import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from matplotlib import colors
import sys
sys.path.append('/vast/home/dmanu/anaconda3/envs/torch_env/lib/python3.8/site-packages')
import torch
import torchvision
from torchvision.transforms import Compose

import sys
sys.path.append('../fwi_ultrasound')
import transforms as T
import network
import classifier
import time



with torch.no_grad():
	#dev = torch.device('cuda')

	model = network.model_dict['FCN4_Deep_Resize_Enc'](upsample_mode='nearest')#.to(dev)
	sd = torch.load('/vast/home/dmanu/Desktop/Ultra_sound/checkpoints/fcn_task/run1/model_2500.pth', map_location=torch.device('cpu') )['model']
	model.encoder.copy_(torch.load('/vast/home/dmanu/Desktop/Ultra_sound/checkpoints/fcn_gauss/run1/model_2500.pth',map_location=torch.device('cpu') )['encoder'])
	model.load_state_dict(sd)
	model.eval()

	classf = classifier.TumorClassifier()#.to(dev)
	sdc = torch.load('models/classifier/checkpoint.pth',map_location=torch.device('cpu') )['model']
	classf.load_state_dict(sdc)
	classf.eval()
	
	cmap = colors.ListedColormap(['black','white','red'])
	bounds = [0, 0.5, 1.5, 2.5]
	norm = colors.BoundaryNorm(bounds,cmap.N)
	#for k in range(34,35):
	for k in range(4):
		print(k)
		count = 0
		seis = np.load('/projects/piml_inversion/ljlozenski/acoustic_measurments/dataset34.npy')#.format(k))
		vmaps = np.load('/projects/piml_inversion/ljlozenski/velocity_maps/dataset34.npy')#.format(k))
		tmaps = classf(torch.from_numpy(vmaps).float()).cpu().detach().numpy() > 0.5#classf(torch.from_numpy(vmaps).to(dev).float()).cpu().detach().numpy() > 0.5
		#tmaps = np.load('tumor_masks/dataset{}.npy'.format(k))
		#vmaps = 2*(vmaps - 1.4)/(1.6 - 1.4) -1
		t1 = time.time()
		for i in range(2):
			up = min(41,7*(i+1))
			seis_small = seis[i*7:up,:,140:,:]
			vmaps_small = vmaps[i*7:up,:,:,:]
			tm_small = tmaps[i*7:up,:,:,:]
			seist = torch.from_numpy(seis_small).float()#.to(dev)
			sgma = 0.0001*0.004*torch.randn(size = seist.size())#, device = seist.get_device())

			outt = model(seist+sgma)
			
			outt += 1
			outt *= (1.6 - 1.4)/2
			outt += 1.4
			learned_tumort = classf(outt) > 0.02
			
			out = outt.cpu().detach().numpy()
			learned_tumor = learned_tumort.cpu().detach()

			for j in range(up-7*i):


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
				plt.savefig('/vast/home/dmanu/Desktop/Ultra_sound/outputs/task_based/out_image{}.png'.format(k*100+count))
				plt.close(f)
				plt.clf()

				f, (a1, a2) = plt.subplots(1,2)
				a1.imshow(tm_small[j,0,:,:], vmin = 0, vmax = 1, cmap = 'gray')
				a1.set_xticklabels([])
				a1.set_xticks([])
				a1.set_yticklabels([])
				a1.set_yticks([])
				a2.imshow(learned_tumor[j,0,:,:]*(2-tm_small[j,0,:,:]), cmap = cmap, norm = norm)
				a2.set_xticklabels([])
				a2.set_xticks([])
				a2.set_yticklabels([])
				a2.set_yticks([])
				f.subplots_adjust(wspace = 0.01)
				plt.savefig('/vast/home/dmanu/Desktop/Ultra_sound/outputs/task_based/tumor_image{}.png'.format(k*100+count))
				plt.close(f)
				plt.clf()
				count += 1
				print(count)
	
