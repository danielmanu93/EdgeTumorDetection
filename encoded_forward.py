import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io

import torch

import sys
sys.path.append('./fwi_ultrasound')

from forward import FWIForward
import transforms as T


if __name__ == "__main__":
	dev = 'cuda'
	devt = torch.device('cuda')
	v_denorm_func = lambda v: v
	s_norm_func = lambda s: s
	model_forward = FWIForward(v_denorm_func, s_norm_func)
	encoder = torch.load('Encoder_matrices/encoder.pt').to(dev)
	with torch.no_grad():


		for k in range(0,19+16):
		#for k in range(5,7):
			print("Starting on Dataset {}".format(k))
			velocity_maps = np.load('velocity_maps/dataset{}.npy'.format(k))
			meas = np.zeros((velocity_maps.shape[0],16,640, 256))
			Nj = np.ceil(velocity_maps.shape[0]/6).astype(int)

			for j in range(Nj):
				torch.cuda.empty_cache()
				ub = np.min([velocity_maps.shape[0],6*(j+1)])
				vel_torch = torch.from_numpy(velocity_maps[6*j:ub,:,:,:]).detach().to(dev)
				meas_torch = model_forward(vel_torch)
				meas_torch = torch.einsum('ij,ajbc->aibc', encoder,meas_torch)
				meas[6*j:ub,:,:,:] = meas_torch.cpu().detach().numpy()
			np.save('acoustic_measurements/dataset{}'.format(k),meas)
			#np.save('small_measurements2/dataset{}'.format(k),meas)
	

	





