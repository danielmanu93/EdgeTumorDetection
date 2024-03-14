import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io

import torch

import sys
sys.path.append('../fwi_ultrasound')

from forward import FWIForward
import transforms as T


if __name__ == "__main__":
	with torch.no_grad():
		dev = 'cuda'
		model_forward = FWIForward(sigma = 2, f = 0.5)
		for k in range(18):
		#for k in range(5,7):
			print("Starting on Dataset {}".format(k))
			if k == 17:
				velocity_maps = np.load('gen_vmaps/test.npy'.format(k))
			else:
				velocity_maps = np.load('gen_vmaps/dataset{}.npy'.format(k))
		
			#velocity_maps = np.load('../vmaps/dataset{}.npy'.format(k))
			meas = np.zeros((velocity_maps.shape[0],64,640, 256))
			Nj = np.ceil(velocity_maps.shape[0]/10).astype(int)
			for j in range(Nj):
				torch.cuda.empty_cache()
				ub = np.min([velocity_maps.shape[0],10*(j+1)])
				vel_torch = torch.from_numpy(velocity_maps[10*j:ub,:,:,:]).detach().to(dev)
				meas_torch = model_forward(vel_torch)
				print(meas_torch.size())
				meas[10*j:ub,:,:,:] = meas_torch.cpu().detach().numpy()
				print(meas.max())
			if k == 17:
				np.save('/projects/piml_inversion/ljlozenski/gen_measurements/test.npy',meas)
			else:
				np.save('/projects/piml_inversion/ljlozenski/gen_measurements/dataset{}'.format(k),meas)
				
				
			#np.save('/projects/piml_inversion/ljlozenski/acoustic_measurements/dataset{}'.format(k),meas)
	

	





