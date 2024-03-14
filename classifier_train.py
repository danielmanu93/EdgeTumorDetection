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
sys.path.append('./fwi_ultrasound')

from forward import FWIForward
from dataset import FWIDataset
import transforms as T
import network
import utils
import classifier

step = 0

def train_one_epoch(model, optimizer, dataloader, device, epoch, print_freq, writer):
	global step
	#model.train()

	# Logger setup
	metric_logger = utils.MetricLogger(delimiter='  ')	
	metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
	metric_logger.add_meter('samples/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))
	header = 'Epoch: [{}]'.format(epoch)

	for data, label in metric_logger.log_every(dataloader, print_freq, header):
		start_time = time.time()
		optimizer.zero_grad()
		data, label = data.to(device), label.to(device)
		output = model(data + 0.1*torch.randn(data.size(), device = device))
		#criterion = nn.CrossEntropyLoss()
		#loss = criterion(output, label)
		#loss = torch.mean( -label*torch.log(1e-4 + output)-(1-label)*torch.log(1e-4 + 1-output))
		#loss = torch.mean( (label - output)**2)/2
		loss = torch.mean(torch.abs(label - output))
		loss.backward()
		optimizer.step()

		loss_val = loss.item()
		batch_size = data.shape[0]
		metric_logger.update(loss=loss_val, lr=optimizer.param_groups[0]['lr'])
		metric_logger.meters['samples/s'].update(batch_size / (time.time() - start_time))
	if writer:
		writer.add_scalar('loss', loss_val, step)
	step += 1


if __name__ == "__main__":

	batch_size = 17
	workers = 16
	sample_ratio = 1
	file_size = 41
	train_anno = '/vast/home/dmanu/Desktop/Ultra_sound/USCT_InversionNet/mask_pairs.txt'
	

	dataset_train = FWIDataset(
		train_anno,
		preload=True,
		sample_ratio=sample_ratio,
		file_size=file_size,
		trunc = False,
		transform_data=None,
		transform_label=None
	)
	train_sampler = RandomSampler(dataset_train)
	
	dataloader_train = DataLoader(
	dataset_train, batch_size=batch_size,
	sampler=train_sampler, num_workers=workers,
        pin_memory=True, drop_last=True, collate_fn=default_collate)
	
	writer = None
	device = 'cuda'

	model = classifier.TumorClassifier().to(device)
	
	Nepochs = 1000
	print_freq = 1

	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	utils.mkdir('models/classifier_gamma_0.1/')
	
	for n in range(Nepochs):
		train_one_epoch(model, optimizer, dataloader_train, device, n, print_freq, writer)
		checkpoint = {
		'model': model.state_dict(),
		'optimizer': optimizer.state_dict(),
		'epoch': n,
		'step': step}
		utils.save_on_master(checkpoint, 'models/classifier_gamma_0.1/checkpoint.pth')
	
		
	

	



