import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import sys
sys.path.append('/vast/home/dmanu/anaconda3/envs/torch_env/lib/python3.8/site-packages')
import torch
import torchvision
from torchvision.transforms import Compose

import sys
sys.path.append('./fwi_ultrasound')
import transforms as T
import network
import classifier


if __name__ == "__main__":
	dev = torch.device('cuda')
	model = classifier.TumorClassifier().to(dev)
	sd = torch.load('/vast/home/dmanu/Desktop/Ultra_sound/USCT_InversionNet/models/classifier_gamma_0.1/checkpoint.pth')['model']
	model.load_state_dict(sd)
	model.eval()
		
	for k in range(1):
		count = 0
		c_np = np.load('/projects/piml_inversion/ljlozenski/velocity_maps/dataset33.npy') #.format(k))
		#c_np = np.load('attenuation_maps/dataset{}.npy'.format(k))
		t_np = np.load('/vast/home/dmanu/Desktop/Ultra_sound/tumor_masks/dataset33.npy') #.format(k))
		for i in range(1):
			c_trch = torch.from_numpy(c_np[10*i:10*(i+1),:,:,:]).to(dev).float()
			t_est = model(c_trch).cpu().detach().numpy()
			#print(t_est.mean())
			#print(t_est.std())
			print(t_est.max())
			print(t_est.min())
			print(t_np.max())
			print(t_np.min())
			for j in range(10):
				f, (a1, a2, a3) = plt.subplots(1,3)

				a1.imshow(c_np[10*i+j,0,:,:], vmin = 1.4, vmax = 1.6, cmap = 'gray')
				a1.set_xticklabels([])
				a1.set_xticks([])
				a1.set_yticklabels([])
				a1.set_yticks([])
				a1.title.set_text('Speed of Sound')

				a2.imshow(t_np[10*i+j,0,:,:], vmin = 0, vmax = 1, cmap = 'gray')
				a2.set_xticklabels([])
				a2.set_xticks([])
				a2.set_yticklabels([])
				a2.set_yticks([])
				a2.title.set_text('True Tumor')

				a3.imshow(t_est[j,0,:,:], vmin = 0, vmax = 1, cmap = 'gray')
				a3.set_xticklabels([])
				a3.set_xticks([])
				a3.set_yticklabels([])
				a3.set_yticks([])
				a3.title.set_text('Learned Tumor')

				f.subplots_adjust(wspace = 0.01)
				plt.savefig('/vast/home/dmanu/Desktop/Ultra_sound/outputs/Classifier_Images/gamma_0.1/out_image{}.png'.format(k*10+count))
				plt.close(f)
				plt.clf()
				count += 1

					
				

	


