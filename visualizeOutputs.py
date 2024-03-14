import numpy as np
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


if __name__ == "__main__":
	option = 'supervised'
	#dev = torch.device('cuda')
	if option == 'supervised':
		model = network.model_dict['FCN4_Deep_Resize_Enc'](upsample_mode='nearest')#.to(dev)
		sd = torch.load('/vast/home/dmanu/Desktop/Ultra_sound/checkpoints/fcn_ultra/run1/checkpoint.pth', map_location=torch.device('cpu'))['model']
		model.load_state_dict(sd)
		model.eval()
		try:
			model.mask = model.mask#.to(dev)
		except:
			pass
		#S = sum(p.numel() for p in model.parameters() if p.requires_grad)
		#print(S)
		#assert False

		#inds = range(35)
		#for k in inds:
		for k in range(1):
			print(k)
			count = 0
			seis = np.load('/vast/home/dmanu/Desktop/Ultra_sound/acoustic_measurements/dataset34.npy')#.format(k))
			vmaps = np.load('/vast/home/dmanu/Desktop/Ultra_sound/velocity_maps/dataset34.npy')#.format(k))
			#vmaps = 2*(vmaps - 1.4)/(1.6 - 1.4) -1
			for i in range(1):
				seis_small = seis[10*i:10*(i+1),:,:,:]
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
					plt.savefig('/vast/home/dmanu/Desktop/Ultra_sound/outputs/out_image{}.png'.format(k*10+count))
					plt.close(f)
					plt.clf()
					count += 1
	else:
		model = network.model_dict['FCN4_Deep_Resize_2'](upsample_mode='nearest', map_location=torch.device('cpu'))#.to(dev)

		sd = torch.load('/vast/home/dmanu/Desktop/Ultra_sound/checkpoints/fcn_ultra/run1/checkpoint.pth')['model']
		model.load_state_dict(sd)
		model.eval()
		try:
			model.mask = model.mask#.to(dev)
		except:
			pass
		
		for k in range(1):
		#for k in range(1):
			print(k)
			count = 0
			#seis = np.load('small_measurements/dataset{}.npy'.format(k))
			seis = np.load('/vast/home/dmanu/Desktop/Ultra_sound/acoustic_measurements/dataset34.npy')
			sf = 1e4
			data_min = -np.log1p(sf*.30)
			data_max =  np.log1p(sf*0.69)
			transform_data = lambda x: x #Compose([T.LogTransform(k=sf),T.MinMaxNormalize(data_min,data_max)])
			
			seis = transform_data(seis)
			#vmaps = np.load('small_maps/dataset{}.npy'.format(k))
			vmaps = np.load('/vast/home/dmanu/Desktop/Ultra_sound/velocity_maps/dataset34.npy')
			for i in range(1):
				seis_small = seis[6*i:6*(i+1),:,:,:]
				vmaps_small = vmaps[6*i:6*(i+1),:,:,:]
				seist = torch.from_numpy(seis_small).float()#.to(dev)

				outt = model(seist)
				out = outt.cpu().detach().numpy()
				out += 1
				out *= (1.6 - 1.4)/2
				out += 1.4
		
			
				for j in range(6):
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
					plt.savefig('/vast/home/dmanu/Desktop/Ultra_sound/outputs/out_image{}.png'.format(k*10+count))
					plt.close(f)
					plt.clf()
					print(count)
					count += 1
				

	


