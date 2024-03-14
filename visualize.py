import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors


def PlotUltrasound(data, data_min, data_max, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    img = plt.imshow(data, cmap="gray", vmin=data_min, vmax=data_max, extent=[0, 256 * 10/1000., 640., 0], interpolation="nearest", aspect="auto")
    for label in  ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(9)
    # ax.set_title('Ultrasound Data',{'family': 'Times New Roman','weight': 'normal','size': 16})
    ax.set_xlabel('Position (km)', fontsize=10)
    ax.set_ylabel('Time (s)', fontsize=10)
    fig.patch.set_facecolor("#C0C0C0") 
    fig.patch.set_edgecolor("#C0C0C0")
    cbar=plt.colorbar(img, ax=ax, shrink=1.0, pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(label='Amplitude', size=10)
    plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
    plt.close('all')


def TaskBasedReconstruction(data, path):
    fig, ax = plt.subplots(figsize=(5,3))
    img = plt.imshow(data, cmap="gray", vmin=1.4, vmax=1.6)
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticklabels([])
    ax.set_yticks([])
    fig.patch.set_facecolor("#C0C0C0") 
    fig.patch.set_edgecolor("#C0C0C0")
    # ax.set_title('Task-Based Reconstruction',{'family': 'Times New Roman','weight': 'normal','size': 2})
    fig.subplots_adjust(wspace = 0.01)
    plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
    plt.close('all')

def TaskBasedTumor(data, path):
    cmap = colors.ListedColormap(['black','white','red'])
    bounds = [0, 0.5, 1.5, 2.5]
    norm = colors.BoundaryNorm(bounds,cmap.N)
    fig, ax = plt.subplots(figsize=(4,3))
    # ax.set_title('Task-Based Tumor',{'family': 'Times New Roman','weight': 'normal','size': 2})
    img = plt.imshow(data, cmap=cmap, norm=norm)
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticklabels([])
    ax.set_yticks([])
    fig.patch.set_facecolor("#C0C0C0") 
    fig.patch.set_edgecolor("#C0C0C0")
    fig.subplots_adjust(wspace = 0.01)
    plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
    plt.close('all')

def Reconstruction(data, path):
    fig, ax = plt.subplots(figsize=(4,3))
    img = plt.imshow(data, cmap="gray", vmin=1.4, vmax=1.6)
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticklabels([])
    ax.set_yticks([])
    fig.patch.set_facecolor("#C0C0C0") 
    fig.patch.set_edgecolor("#C0C0C0")
    # ax.set_title('Reconstruction',{'family': 'Times New Roman','weight': 'normal','size': 8})
    fig.subplots_adjust(wspace = 0.01)
    plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
    plt.close('all')

def ReconstructionTumor(data, path):
    cmap = colors.ListedColormap(['black','white','red'])
    bounds = [0, 0.5, 1.5, 2.5]
    norm = colors.BoundaryNorm(bounds,cmap.N)
    fig, ax = plt.subplots(figsize=(5,3))
    # ax.set_title('Reconstruction Tumor',{'family': 'Times New Roman','weight': 'normal','size': 2})
    img = plt.imshow(data, cmap=cmap, norm=norm)
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticklabels([])
    ax.set_yticks([])
    fig.patch.set_facecolor("#C0C0C0") 
    fig.patch.set_edgecolor("#C0C0C0")
    fig.subplots_adjust(wspace = 0.01)
    plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
    plt.close('all')