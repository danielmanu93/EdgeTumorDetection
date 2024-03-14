import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import scipy.io as io
from scipy.ndimage.measurements import label

import torch

import sys
sys.path.append('../fwi_ultrasound')
from born_forward import BornForward

import network
import classifier


if __name__ == "__main__":
	#dev = torch.device('cuda')
	#model_names = ['gauss_encoder.pth', 'gamma001.pth', 'gamma01.pth', 'gamma1.pth', 'gamma_one.pth', 'gamma10.pth','gamma_infty.pth']
	model_names = ['checkpoint.pth', 'subsample.pth', 'gauss_encoder.pth', 'trained_encoder.pth']

	tumor_classifier = classifier.TumorClassifier()#.to(dev)
	class_sd = torch.load('models/classifier/checkpoint.pth', map_location=torch.device('cpu'))['model']
	tumor_classifier.load_state_dict(class_sd)
	tumor_classifier.eval()
	detection_thresh = 0.02

	for model_name in model_names:
		err = []
		SSIM = []
		dice = []

		try:
			model = network.model_dict['FCN4_Deep_Resize_Enc'](upsample_mode='nearest', init = None)#.to(dev)
			model.encoder = torch.load('models/fcn/' + model_name, map_location=torch.device('cpu'))['encoder']
		except:
			model = network.model_dict['FCN4_Deep_Resize_2'](upsample_mode='nearest')#.to(dev)
		sd = torch.load('models/fcn/' + model_name,  map_location=torch.device('cpu'))['model']
		model.load_state_dict(sd)
		model.eval()
		inds = np.arange(34,35)
		struct = np.ones((3,3), dtype = np.int)

		
		
		for k in inds:
			count = 0
			seis = np.load('/projects/piml_inversion/ljlozenski/hf_waveoffset/dataset{}.npy'.format(k))
			vmaps = np.load('../velocity_maps/dataset{}.npy'.format(k))
			tmaps = tumor_classifier(torch.from_numpy(vmaps).float()).cpu().detach().numpy() > 0.5
			#tmaps = np.load('tumor_masks/dataset{}.npy'.format(k))
			for i in range(6):
				mx = min([7*(i+1),41])
				seis_small = seis[7*i:mx,:,140:,:]
				vmaps_small = vmaps[7*i:mx,:,:,:]
				tmaps_small = tmaps[7*i:mx,:,:,:]
				seist = torch.from_numpy(seis_small).float()#.to(dev)
				sgma = 0.001*0.004*torch.randn(size = seist.size())#, device = dev)
				print(seis.std())

				outt = 0.1*(model(seist+sgma)+1)+1.4
				outt = outt 
				tt_recon = tumor_classifier(outt)
				out = outt.cpu().detach().numpy()
				t_recon = tt_recon.cpu().detach().numpy() > detection_thresh
				for j in range(mx-7*i):
					SSIM.append(ssim(vmaps_small[j,0,:,:], out[j,0,:,:], data_range = 0.2))
					err.append(np.linalg.norm( out[j,0,:,:]-vmaps_small[j,0,:,:]))

					l, n0 = label(t_recon[j,0,:,:], struct)
					l, n1 = label(tmaps_small[j,0,:,:], struct)
					dice_den = n0+n1
					if dice_den > 0:
						l, n2 = label(t_recon[j,0,:,:]*tmaps_small[j,0,:,:], struct)
						dice.append( 2*n2/dice_den)
					else:
						dice.append(1.)

		

		print(model_name)
		print("		Average Training Error = ", np.array(err[:-41]).mean())
		print("		Training Error Standard Deviation = ", np.array(err[:-41]).std())
		print("		Average Training SSIM = ", np.array(SSIM[:-41]).mean())
		print("		Training SSIM Standard Deviation = ", np.array(SSIM[:-41]).std())
		print("		Average Training Dice = ", np.array(dice[:-41]).mean())
		print("		Training Dice Standard Deviation = ", np.array(dice[:-41]).std())
		print('\n')
		print("		Average Testing Error = ", np.array(err[-41:]).mean())
		print("		Testing Error Standard Deviation = ", np.array(err[-41:]).std())
		print("		Average Testing SSIM = ", np.array(SSIM[-41:]).mean())
		print("		Testing Error Standard SSIM= ", np.array(SSIM[-41:]).std())
		print("		Average Testing Dice = ", np.array(dice[-41:]).mean())
		print("		Testing Dice Standard Deviation = ", np.array(dice[-41:]).std())
		np.save('Errors/' + model_name[:-4] + '_rmse', err)
		np.save('Errors/' + model_name[:-4] + '_ssim,', SSIM)
		np.save('Errors/' + model_name[:-4] + '_dice', dice)
				
					

		


