import datetime
import os
import sys
sys.path.append('/vast/home/dmanu/anaconda3/envs/torch_env/lib/python3.8/site-packages')
import time
import torch
from torch import nn
import numpy as np
import torchvision
from torchvision.transforms import Compose
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
sys.path.append('../fwi_ultrasound')

from forward import FWIForward
from dataset import FWIDataset
import transforms as T
import network
import classifier

sys.path.append('/projects/piml_inversion/hwang/repo/UPFWI/src/')
import utils
from scheduler import WarmupMultiStepLR
from functools import reduce
import operator

step = 0

def train_one_epoch(model, criterion, optimizer, lr_scheduler, 
                    dataloader, device, epoch, print_freq, writer):
    global step
    model.train()

    # Logger setup
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('samples/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for data, label in metric_logger.log_every(dataloader, print_freq, header):
        start_time = time.time()
        optimizer.zero_grad()
        data, label = data.to(device), label.to(device)
        nrots = 0 #np.random.randint(4)
        nflip = 0 #np.random.randint(2)

        n = data.size(1)
        N = data.size(3)

        if nrots > 0:
                data = torch.roll(data,nrots*int(n/4), dims = 1)
                data = torch.roll(data,nrots*int(N/4),dims = 3)
                label = torch.rot90(label, k = nrots, dims = [2, 3])
        if nflip > 0:
                data = data[:,-np.arange(n),:,:]
                data = data[:,:,:,-np.arange(N)]
                label = torch.flip(label,dims = (2,))


        sgma = 0.0001*data.std()*torch.randn(size = data.size(), device = data.get_device())
        output = model(data + sgma)
        loss, loss_c2v, loss_g2v = criterion(output, label)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_c2v_val = loss_c2v.item()
        loss_g2v_val = loss_g2v.item()
        batch_size = data.shape[0]
        metric_logger.update(loss=loss_val, loss_c2v=loss_c2v_val, 
            loss_g2v=loss_g2v_val, lr=optimizer.param_groups[0]['lr'])
        metric_logger.meters['samples/s'].update(batch_size / (time.time() - start_time))
        if writer:
            writer.add_scalar('loss', loss_val, step)
            writer.add_scalar('loss_c2v', loss_c2v_val, step)
            writer.add_scalar('loss_g2v', loss_g2v_val, step)
        step += 1

    return loss_val


def evaluate(model, criterion, dataloader, device, writer):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = 'Test:'
    with torch.no_grad():
        for data, label in metric_logger.log_every(dataloader, 20, header):
            data = data.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            output = model(data)
            loss, loss_c2v, loss_g2v = criterion(output, label)
            batch_size = data.shape[0]
            metric_logger.update(loss=loss.item(), 
                loss_c2v=loss_c2v.item(), 
                loss_g2v=loss_g2v.item())

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(' * Loss {loss.global_avg:.8f}\n'.format(loss=metric_logger.loss))
    if writer:
        writer.add_scalar('loss', metric_logger.loss.global_avg, step)
        writer.add_scalar('loss_c2v', metric_logger.loss_c2v.global_avg, step)
        writer.add_scalar('loss_g2v', metric_logger.loss_g2v.global_avg, step)

    return metric_logger.loss.global_avg


def main(args):
    global step

    utils.mkdir(args.output_path)
    train_writer, val_writer = None, None
    utils.init_distributed_mode(args)
    if args.tensorboard:
        if not args.distributed or (args.rank == 0) and (args.local_rank == 0):
            train_writer = SummaryWriter(os.path.join(args.output_path, 'logs', 'train'))
            val_writer = SummaryWriter(os.path.join(args.output_path, 'logs', 'val'))
                                                                    
    print(args)
    print('torch version: ', torch.__version__)
    print('torchvision version: ', torchvision.__version__)

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True


    label_min = 1.4
    label_max =1.6
    data_min = -0.30
    data_max = 0.69

    

    # Data loading code
    print('Loading data')
    print('Loading training data')
    
    # Normalize data and label to [-1, 1]

    transform_data = None
    """transform_data = Compose([
        T.MinMaxNormalize(data_min,data_max),T.LogTransform(k=args.k) # (legacy) log transformation
    ])"""
    transform_label = Compose([
        T.MinMaxNormalize(label_min, label_max)
    ])
    if args.train_anno[-3:] == 'txt':
        dataset_train = FWIDataset(
            args.train_anno,
            preload=True,
            sample_ratio=args.sample_ratio,
            file_size=args.file_size,
            transform_data=transform_data,
            transform_label=transform_label
        )
    else:
        dataset_train = torch.load(args.train_anno)

    print('Loading validation data')
    if args.val_anno[-3:] == 'txt':
        dataset_valid = FWIDataset(
            args.val_anno,
            preload=True,
            sample_ratio=args.sample_ratio,
            file_size=args.file_size,
            transform_data=transform_data,
            transform_label=transform_label
        )
    else:
        dataset_valid = torch.load(args.val_anno)

    print('Creating data loaders')
    if args.distributed:
        train_sampler = DistributedSampler(dataset_train, shuffle=True)
        valid_sampler = DistributedSampler(dataset_valid, shuffle=True)
    else:
        train_sampler = RandomSampler(dataset_train)
        valid_sampler = RandomSampler(dataset_valid)

    dataloader_train = DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        pin_memory=True, drop_last=True, collate_fn=default_collate)

    dataloader_valid = DataLoader(
        dataset_valid, batch_size=args.batch_size,
        sampler=valid_sampler, num_workers=args.workers,
        pin_memory=True, collate_fn=default_collate)

    print('Creating model')
    
    if args.model not in network.model_dict:
        print('Unsupported model.')
        sys.exit()
         
    if args.up_mode:    
        model = network.model_dict[args.model](upsample_mode=args.up_mode).to(device)
    else:
        model = network.model_dict[args.model]().to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    l2loss = nn.MSELoss()
    tumor_classifier = classifier.TumorClassifier().to(device)
    class_sd = torch.load('models/classifier/checkpoint.pth')['model']
    tumor_classifier.load_state_dict(class_sd)
    tumor_classifier.eval()



    def criterion(pred, gt):
        g1, g2, g3, g4 = tumor_classifier.expand(0.1*(gt + 1) + 1.4)
        p1, p2, p3, p4 = tumor_classifier.expand(0.1*(pred + 1) + 1.4)
        loss_c2v = l2loss(p1,g1) + l2loss(p2,g2) + l2loss(p3,g3) + l2loss(p4,g4)
        loss_g2v = l2loss(pred, gt)
        loss = args.lambda_c2v * loss_c2v + args.lambda_g2v * loss_g2v
        return loss, loss_c2v, loss_g2v
    
    # Scale lr according to effective batch size
    lr = args.lr * args.world_size 
    #optimizer = torch.optim.SGD(
    #    model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    lr_scheduler = ReduceLROnPlateau(optimizer, patience = 10000, verbose=True)

    model_without_ddp = model
    if args.distributed:
 
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        model_without_ddp = model.module

    if args.resume: # load from checkpoint
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        with torch.no_grad():
                model.encoder.copy_(checkpoint['encoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        #lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        #args.start_epoch = checkpoint['epoch'] + 1
        #step = checkpoint['step']
        print('Loaded Checkpoint')

    train_loss, valid_loss = [], []
    print('Start training')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        loss = train_one_epoch(model, criterion, optimizer, lr_scheduler, dataloader_train,
                        device, epoch, args.print_freq, train_writer)
        #print(loss)
        val_loss = evaluate(model, criterion, dataloader_valid, device, val_writer)
        #print(val_loss)
        lr_scheduler.step(val_loss)
        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'encoder': model.encoder,
            'step': step,
            'args': args}
        # Save checkpoint per epoch
        torch.save(
            checkpoint,
            os.path.join(args.output_path, 'checkpoint.pth'))
        # Save checkpoint every epoch block
        if args.output_path and (epoch + 1) % args.epoch_block == 0:
            torch.save(
                checkpoint,
                os.path.join(args.output_path, 'model_{}.pth'.format(epoch + 1)))
            
        train_loss.append(loss)
        valid_loss.append(val_loss)

    train_loss_array = np.array(train_loss)
    valid_loss_array = np.array(valid_loss)
        
    np.savetxt("/vast/home/dmanu/Desktop/Ultra_sound/train loss.txt", train_loss_array)
    np.savetxt("/vast/home/dmanu/Desktop/Ultra_sound/validation loss.txt", valid_loss_array)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='FCN Training')
    parser.add_argument('--anno-path', default='USCT_InversionNet', help='dataset files location')
    parser.add_argument('-t', '--train-anno', default='/vast/home/dmanu/Desktop/Ultra_sound/USCT_InversionNet/train_data.txt', help='name of train anno')
    parser.add_argument('-v', '--val-anno', default='/vast/home/dmanu/Desktop/Ultra_sound/USCT_InversionNet/test_data.txt', help='name of val anno')
    parser.add_argument('-fs', '--file-size', default=41, type=int, help='samples per data file')
    #parser.add_argument('-fs', '--file-size', default=2, type=int, help='samples per data file')
    parser.add_argument('-ds', '--dataset', default='flat', type=str, help='dataset option for normalization')
    parser.add_argument('-o', '--output-path', default='models', help='path where to save')
    parser.add_argument('-n', '--save-name', default='fcn', help='saved name for this run')
    parser.add_argument('-s', "--suffix", type=str, default=None)
    parser.add_argument('-m', '--model', help='select inverse model')
    parser.add_argument('--up_mode', default=None, help='upsample mode')
    parser.add_argument('-d', '--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=17, type=int)
    #parser.add_argument('-b', '--batch-size', default=2, type=int)
    parser.add_argument('-sr', '--sample_ratio', type=int, default=1, help='subsample ratio of data')
    parser.add_argument('-eb', '--epoch_block', type=int, default=500, help='epochs in a saved block')
    parser.add_argument('-nb', '--num_block', type=int, default=5, help='number of saved block')
    parser.add_argument('-j', '--workers', default=16, type=int, help='number of data loading workers (default: 16)')
    #parser.add_argument('-j', '--workers', default=2, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--k', default=1e4, type=float, help='k in log transformation')
    parser.add_argument('-mo', '--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('-wd', '--weight-decay', default=0, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    #parser.add_argument('--resume', default='models/fcn/checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('-c2v', '--lambda_c2v', type=float, default= 1.0)
    parser.add_argument('-g2v', '--lambda_g2v', type=float, default= 0.0)
    
    # distributed training parameters
    parser.add_argument('--sync-bn', action='store_true', help='Use sync batch norm')
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    # tensorboard
    parser.add_argument('--tensorboard', action='store_true', help='Use tensorboard for logging.')
    args = parser.parse_args()

    args.output_path = os.path.join(args.output_path, args.save_name, args.suffix or '')
    args.train_anno = os.path.join(args.anno_path, args.train_anno)
    args.val_anno = os.path.join(args.anno_path, args.val_anno)
    
    args.epochs = args.epoch_block * args.num_block

    return args


if __name__ == '__main__':

    args = parse_args()
    main(args)
