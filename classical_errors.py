import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import scipy.io as io
from scipy.ndimage.measurements import label
from matplotlib import colors

import torch

import sys
sys.path.append('../fwi_ultrasound')

import network
import classifier


if __name__ == "__main__":
	#dev = torch.device('cuda')
	tumor_classifier = classifier.TumorClassifier()
	class_sd = torch.load('models/classifier/checkpoint.pth', map_location=torch.device('cpu'))['model']
	tumor_classifier.load_state_dict(class_sd)
	tumor_classifier.eval()
	
	true_sos = np.load('../velocity_maps/dataset34.npy')
	#true_tumor = np.load('tumor_masks/dataset34.npy')
	true_tumor = tumor_classifier(torch.from_numpy(true_sos).float()).cpu().detach().numpy() > 0.5	
	recons = np.load('classic_test_results_rob.npy')
	tumor_recon = tumor_classifier(torch.from_numpy(recons).float()).cpu().detach().numpy() > 0.02

	SSIMS = []
	RMSES = []
	DICE = []
	struct = np.ones((3,3), dtype = np.int)
	cmap = colors.ListedColormap(['black','white','red'])
	bounds = [0, 0.5, 1.5, 2.5]
	norm = colors.BoundaryNorm(bounds,cmap.N)
	for j in range(41):
		SSIMS.append(ssim(true_sos[j,0,:,:],recons[j,0,:,:], data_range = 0.2))
		RMSES.append(np.linalg.norm(true_sos[j,0,:,:]-recons[j,0,:,:]))

		l, n0 = label(tumor_recon[j,0,:,:], struct)
		l, n1 = label(true_tumor[j,0,:,:], struct)
		dice_den = n0+n1
		if dice_den > 0:
			l, n2 = label(tumor_recon[j,0,:,:]*true_tumor[j,0,:,:], struct)
			DICE.append( 2*n2/dice_den)
		else:
			DICE.append(1.)
		f, (a1, a2) = plt.subplots(1,2)
		a1.imshow(true_sos[j,0,:,:], vmin = 1.4, vmax = 1.6, cmap = 'gray')
		a1.set_xticklabels([])
		a1.set_xticks([])
		a1.set_yticklabels([])
		a1.set_yticks([])
		a2.imshow(recons[j,0,:,:], vmin = 1.4, vmax = 1.6, cmap = 'gray')
		a2.set_xticklabels([])
		a2.set_xticks([])
		a2.set_yticklabels([])
		a2.set_yticks([])
		f.subplots_adjust(wspace = 0.01)
		plt.savefig('Classic_Recons/out_image{}.png'.format(j))
		plt.close(f)
		plt.clf()

		f, (a1, a2) = plt.subplots(1,2)
		a1.imshow(true_tumor[j,0,:,:], vmin = 0, vmax = 1, cmap = 'gray')
		a1.set_xticklabels([])
		a1.set_xticks([])
		a1.set_yticklabels([])
		a1.set_yticks([])
		a2.imshow(tumor_recon[j,0,:,:]*(2-true_tumor[j,0,:,:]), cmap = cmap, norm = norm)
		a2.set_xticklabels([])
		a2.set_xticks([])
		a2.set_yticklabels([])
		a2.set_yticks([])
		f.subplots_adjust(wspace = 0.01)
		plt.savefig('Classic_Recons/tumor_image{}.png'.format(j))
		plt.close(f)
		plt.clf()
	print("FWI Results")


	print("		Average Testing Error = ", np.array(RMSES).mean())
	print("		Testing Error Standard Deviation = ", np.array(RMSES).std())
	print("		Average Testing SSIM = ", np.array(SSIMS).mean())
	print("		Testing Error Standard SSIM= ", np.array(SSIMS).std())
	print("		Average Testing Dice = ", np.array(DICE).mean())
	print("		Testing Dice Standard Deviation = ", np.array(DICE).std())
	np.save('Errors/FWI_rmse', RMSES)
	np.save('Errors/FWI_ssim,', SSIMS)
	np.save('Errors/FWI_dice', DICE)
		
		
		




	
		
		
	
