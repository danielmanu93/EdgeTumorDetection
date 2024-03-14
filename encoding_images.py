import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as io
import sys
sys.path.append('/vast/home/dmanu/anaconda3/envs/torch_env/lib/python3.8/site-packages')
import torch
import torchvision
from torchvision.transforms import Compose

import sys
sys.path.append('../fwi_ultrasound')
import transforms as T
import network
import os


with torch.no_grad():
	#dev = torch.device('cuda')


	model = network.model_dict['FCN4_Deep_Resize_Enc'](upsample_mode='nearest', init = None)#.to(dev)
	#model = network.model_dict['FCN4_Deep_Resize_2'](upsample_mode='nearest')#.to(dev)
	sd = torch.load('/vast/home/dmanu/Desktop/Ultra_sound/checkpoints/fcn_trained_encoder/run2/model_2500.pth', map_location=torch.device('cpu'))['model']
	model.load_state_dict(sd)
	model.encoder.copy_(torch.load('/vast/home/dmanu/Desktop/Ultra_sound/checkpoints/fcn_trained_encoder/run2/model_2500.pth', map_location=torch.device('cpu'))['encoder'])
	
	
	#out_folder = 'Gauss_encoder/'
	out_folder = '/vast/home/dmanu/Desktop/Ultra_sound/outputs/trained_encoder/'
	#out_folder = 'SingleShotResults/'
	#out_folder = 'Trained_encoder/'
	try:
		os.mkdir(out_folder)
	except:
		pass
	try:
		E = model.encoder.cpu().detach().numpy()
		plt.hist(E.flatten(), bins = 100)		
		plt.savefig(out_folder + 'encoder_dist.png')
		plt.clf()
		plt.imshow(E)
		plt.savefig(out_folder +'encoder.png')
		plt.clf()
	except:
		 pass
	
	inds = np.arange(34)#[0,1,2,3,4,5,6,7,8,9,10,11,12,13,24,15,16,17,23]

	model.eval()

	
	#for k in inds:
	for k in range(1):
		print(k)
		count = 0
		seis = np.load('/projects/piml_inversion/ljlozenski/acoustic_measurments/dataset34.npy') #.format(k))
		vmaps = np.load('/projects/piml_inversion/ljlozenski/velocity_maps/dataset34.npy') #.format(k))
		#vmaps = 2*(vmaps - 1.4)/(1.6 - 1.4) -1
		for i in range(1):
			seis_small = seis[10*i:10*(i+1),:,140:,:]
			vmaps_small = vmaps[10*i:10*(i+1),:,:,:]
			seist = torch.from_numpy(seis_small).float()#.to(dev)
			outt = model(seist)
			out = outt.cpu().detach().numpy()
			out += 1
			out *= (1.6 - 1.4)/2
			out += 1.4
	
		
			for j in range(10):
				f, (a1, a2) = plt.subplots(1,2)
				a1.imshow(vmaps_small[j,0,:,:], vmin = 1.4, vmax = 1.6, cmap = 'gray')
				a1.set_xticklabels([])
				a1.set_xticks([])
				a1.set_yticklabels([])
				a1.set_yticks([])
				a2.imshow(out[j,0,:,:], vmin = 1.4, vmax = 1.6, cmap = 'gray')
				a2.set_xticklabels([])
				a2.set_xticks([])
				a2.set_yticklabels([])
				a2.set_yticks([])
				f.subplots_adjust(wspace = 0.01)
				#plt.savefig('Random_encoder/out_image{}.png'.format(k*10+count))
				plt.savefig(out_folder + 'out_image{}.png'.format(k*10+count))
				plt.close(f)
				plt.clf()
				count += 1

			
				

	


