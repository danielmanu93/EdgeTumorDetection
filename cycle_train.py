import datetime
import os
import sys
import time
import torch
import numpy as np
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import Compose

import sys
sys.path.append('./fwi_ultrasound')

from forward import FWIForward
from dataset import FWIDataset
import transforms as T
import network

sys.path.append('/projects/piml_inversion/hwang/repo/UPFWI/src/')
import utils
from scheduler import WarmupMultiStepLR

step = 0

def train_one_epoch(model, model_forward,  optimizer, lr_scheduler, criterion,
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

        pred_v = model(data)
        recon_s = model_forward(pred_v)
        
        #loss, loss_part = criterion(data, label, pred_v=pred_v, recon_s=recon_s)
        loss, loss_part = criterion(data, label, pred_v=None, recon_s=recon_s)
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        batch_size = data.shape[0]
        metric_logger.update(loss=loss_val, **loss_part, lr=optimizer.param_groups[0]['lr'])
        metric_logger.meters['samples/s'].update(batch_size / (time.time() - start_time))
        if writer:
            # writer.add_scalar('loss', loss_part['loss_g1v'] + loss_part['loss_g2v'], step)
            writer.add_scalar('loss', loss_val, step)
            for k, v in loss_part.items():
                writer.add_scalar(k, v, step)
        step += 1
        lr_scheduler.step()
        # return


def evaluate(model, model_forward, criterion, dataloader, device, writer):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = 'Test:'
    with torch.no_grad():
        for data, label in metric_logger.log_every(dataloader, 20, header):
            data = data.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            pred_v = model(data)
            recon_s = model_forward(pred_v)

            loss, loss_part = criterion(data, label, pred_v=pred_v, recon_s=recon_s)

            metric_logger.update(loss=loss.item(), **loss_part)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()

    for k in loss_part.keys():
        loss_part[k] = getattr(metric_logger, k).global_avg

    print(' * Loss {loss.global_avg:.8f}\n'.format(loss=metric_logger.loss))
    for k, v in loss_part.items():
        print('  - {name} {loss:.8f}\n'.format(name=k.capitalize(), loss=v))

    if writer:
        # writer.add_scalar('loss', loss_part['loss_g1v'] + loss_part['loss_g2v'], step)
        writer.add_scalar('loss', loss.item(), step)
        for k, v in loss_part.items():
            writer.add_scalar(k, v, step)

    return metric_logger.loss.global_avg, loss_part


def main(args):
    global step

    utils.mkdir(args.output_path)
    train_writer, val_writer = None, None
    utils.init_distributed_mode(args)

    if args.tensorboard:
        if not args.distributed or (args.rank == 0 and args.local_rank == 0):
            train_writer = SummaryWriter(os.path.join(args.output_path, 'logs', 'train'))
            val_writer = SummaryWriter(os.path.join(args.output_path, 'logs', 'val'))
                                                                    
    print(args)
    print('torch version: ', torch.__version__)
    print('torchvision version: ', torchvision.__version__)

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    label_min = 1.4
    label_max = 1.6
    data_min = -np.log1p(args.k*.30)
    data_max =  np.log1p(args.k*.69)

    # Data loading code
    print('Loading data')
    print('Loading training data')

    
    # Normalize data and label to [-1, 1]
    transform_data = lambda x: x
	#Compose([T.LogTransform(k=args.k),T.MinMaxNormalize(data_min,data_max) # (legacy) log transformation ])
    transform_label = Compose([
        T.MinMaxNormalize(label_min, label_max)
    ])

    if args.train_anno[-3:] == 'txt':
        dataset_train = FWIDataset(
            args.train_anno,
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
    model.mask = model.mask.to(device)

    # Norm/denorm function for forward modeling
    v_denorm_func = lambda v: T.minmax_denormalize(v, label_min, label_max)
    s_log_func = lambda s: T.log_transform_tensor(s, k=args.k)#T.minmax_normalize(s_log_func(s), log_data_min, log_data_max)
    s_scale_func = lambda s: T.minmax_normalize(s_log_func(s), data_min, data_max)#log_transform_tensor(s, k=args.k)
    #s_norm_func = lambda s: T.RootTransform(data_lb, data_ub, sf)
    model_forward = FWIForward(v_denorm_func, s_scale_func)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.perceptual_loss:
        vgg = utils.VGGPerceptualLoss().to(device)
        cyc = utils.CycleLoss(args)
        if args.distributed:
            vgg = DistributedDataParallel(vgg, device_ids=[args.local_rank], find_unused_parameters=True)
        def criterion(data, label, pred_v, recon_s):
            loss, loss_part = cyc(data, label, pred_v=pred_v, recon_s=recon_s)
            loss_p1s, loss_p2s = vgg(recon_s, data, feature_layers=args.feature_layers)
            loss += args.lambda_p1s * loss_p1s + args.lambda_p2s * loss_p2s
            loss_part['loss_p1s'] = loss_p1s
            loss_part['loss_p2s'] = loss_p2s
            return loss, loss_part
    else:            
        criterion = utils.CycleLoss(args)

    # Scale lr according to effective batch size
    lr = args.lr * args.world_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    warmup_iters = args.lr_warmup_epochs * len(dataloader_train)
    lr_milestones = [len(dataloader_train) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(
        optimizer, milestones=lr_milestones, gamma=args.lr_gamma,
        warmup_iters=warmup_iters, warmup_factor=1e-5)

    model_without_ddp = model
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        step = checkpoint['step']
        lr_scheduler.milestones=lr_milestones

    print('Start training')

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, model_forward, optimizer, lr_scheduler, criterion, 
                        dataloader_train, device, epoch, args.print_freq, train_writer)
        evaluate(model, model_forward, criterion, dataloader_valid, device, val_writer)
        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'step': step,
            'args': args}
        # Save checkpoint per epoch
        utils.save_on_master(
            checkpoint,
            os.path.join(args.output_path, 'checkpoint.pth'))
        # Save checkpoint every epoch block
        if args.output_path and (epoch + 1) % args.epoch_block == 0:
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_path, 'model_{}.pth'.format(epoch + 1)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='CycleFCN Training')
    parser.add_argument('--anno-path', default='', help='dataset files location')
    parser.add_argument('-ds', '--dataset', default='25', type=str, help='dataset option for forward parameters')
    #parser.add_argument('-t', '--train-anno', default='small_data.txt', help='name of train anno')
    #parser.add_argument('-v', '--val-anno', default='small_test.txt', help='name of val anno')
    parser.add_argument('-t', '--train-anno', default='one_data.txt', help='name of train anno')
    parser.add_argument('-v', '--val-anno', default='one_data.txt', help='name of val anno')
    #parser.add_argument('-fs', '--file-size', default=779, type=int, help='samples per data file')
    parser.add_argument('-fs', '--file-size', default=6, type=int, help='samples per data file')
    parser.add_argument('-o', '--output-path', default='models', help='path where to save')
    parser.add_argument('-n', '--save-name', default='cyclefcn', help='saved name for this run')
    parser.add_argument('-s', '--suffix', type=str, default=None)
    parser.add_argument('-m', '--model', help='select inverse model')
    parser.add_argument('--up_mode', default=None, help='upsample mode')
    parser.add_argument('--mode', default=1, type=int, help='forward calculate mode')
    parser.add_argument('-d', '--device', default='cuda', help='device')
    parser.add_argument('-sr', '--sample_ratio', type=int, default=1, help='subsample ratio of data')
    #parser.add_argument('-b', '--batch-size', default=7, type=int)
    parser.add_argument('-b', '--batch-size', default=6, type=int)
    parser.add_argument('-eb', '--epoch_block', type=int, default=300, help='epochs in a saved block')
    parser.add_argument('-nb', '--num_block', type=int, default=4, help='number of saved block')
    parser.add_argument('-j', '--workers', default=16, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--k', default=1, type=float, help='k in log transformatfion')
    parser.add_argument('-mo', '--momentum', default=0.0, type=float, help='momentum')
    parser.add_argument('-wd', '--weight-decay', default=0 , type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('-lm', '--lr-milestones', nargs='+', default=[], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=0, type=int, help='number of warmup epochs')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    #parser.add_argument('--resume', default='models/cyclefcn/checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
    parser.add_argument('-g1v', '--lambda_g1v', type=float, default=0.0)
    parser.add_argument('-g2v', '--lambda_g2v', type=float, default=0.0)
    parser.add_argument('-g1s', '--lambda_g1s', type=float, default=0.0)
    parser.add_argument('-g2s', '--lambda_g2s', type=float, default=0.0)
    parser.add_argument('-c1v', '--lambda_c1v', type=float, default=0.0)
    parser.add_argument('-c2v', '--lambda_c2v', type=float, default=0.0)

    parser.add_argument('-c1s', '--lambda_c1s', type=float, default=0.0)
    parser.add_argument('-c2s', '--lambda_c2s', type=float, default=1.0)

    parser.add_argument('-p1s', '--lambda_p1s', type=float, default=0.0)
    parser.add_argument('-p2s', '--lambda_p2s', type=float, default=0.0)
    parser.add_argument('-ploss', '--perceptual_loss', action='store_true', help='use perceptual loss')
    parser.add_argument('-fl', '--feature_layers', nargs='+', default=[1], type=int, 
        help='index of layers used for perceptual loss')

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
    m = np.load('small_measurements/dataset0.npy')
    m = m[:600:,:,:]
    np.save('one_measurement',m)

    m = np.load('small_maps/dataset0.npy')
    m = m[:600,:,:,:]
    np.save('one_map',m)

    args = parse_args()
    main(args)
