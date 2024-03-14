import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import time

class FWIForward(nn.Module):
    def __init__(self, transducer_arc_width = 160, n_transducers = 64, nmeas = 256, nx = 360, dx = 0.6 , nt = 640, dt= 0.2, sigma = 12, tc = 3.2, f = 1):
        super(FWIForward, self).__init__()
        self.device = torch.device('cuda')
        self.arc = transducer_arc_width
        self.NT = n_transducers
        thetas = np.linspace(0,2*np.pi,nmeas+1)[:-1]  #+ np.pi/nmeas
	

        transducer_inds = np.floor( self.arc* np.stack((np.cos(thetas), np.sin(thetas))) + (nx-1)/2).astype(int)
        self.meas_inds = nx*transducer_inds[0,:] + transducer_inds[1,:]
        self.in_inds = self.meas_inds[::int(nmeas/self.NT)]
	

        self.dx = dx
        self.nx = nx
        self.dt = dt
        self.nt = nt
        self.sigma = sigma
        self.tc = tc
        self.fc = f

        t = np.arange(self.nt)*self.dt
        self.source = np.exp(-(t-self.tc)**2/(2*self.sigma**2))*np.sin(2*np.pi*self.fc*t)#/(dx**2)

	

    def get_Abc(self, vp, nbc = 15):
        dx = self.dx
        dimrange = 1.0*torch.unsqueeze(torch.arange(nbc, device=self.device), dim=-1)
        damp = torch.zeros_like(vp, device=self.device, requires_grad=False) #
        
        velmin,_ = torch.min(vp.view(vp.shape[0],-1), dim=-1, keepdim=False)

        nzbc, nxbc = self.nx, self.nx
        nz = nzbc-2*nbc
        nx = nxbc-2*nbc
        a = (nbc-1)*dx
        
        kappa = 3.0 * velmin * np.log(1e4) / (2.0 * a)
        kappa = torch.unsqueeze(kappa,dim=0)
        kappa = torch.repeat_interleave(kappa, nbc, dim=0)
        
        damp1d = kappa * (dimrange*dx/a) ** 2
        damp1d = damp1d.permute(1,0)#.unsqueeze(1)
        damp[:,:nbc, :] = torch.repeat_interleave(torch.flip(damp1d,dims=[-1]).unsqueeze(-1), vp.shape[-1], dim=-1) 
        damp[:,-nbc:,:] = torch.repeat_interleave(damp1d.unsqueeze(-1), vp.shape[-1], dim=-1) 
        damp[:,:, :nbc] = torch.repeat_interleave(torch.flip(damp1d,dims=[-1]).unsqueeze(-2), vp.shape[-2], dim=-2) 
        damp[:,:,-nbc:] = torch.repeat_interleave(damp1d.unsqueeze(-2), vp.shape[-2], dim=-2) 
        return damp
	


    def forward(self, v):
        v = F.pad(v, [73, 73, 73, 73], value = 1.5)
        v = v[:,0,:,:]
        kappa = self.get_Abc(v)*self.dt
        v = v**2
        alpha = v*(self.dt/self.dx) ** 2
        beta = self.dt**2*v.reshape((v.shape[0],v.shape[1]*v.shape[2]))[:,self.in_inds]
	
        c1 = -2.5
        c2 = 4/3
        c3 = - 1/12

        temp1 = 2+2*c1*alpha-kappa
        temp0 = 1-kappa

        meas = []
        p1 = torch.zeros((v.shape[0], self.NT, v.shape[1], v.shape[2]), device=self.device, requires_grad=True)
        p0 = torch.zeros((v.shape[0], self.NT, v.shape[1], v.shape[2]), device=self.device, requires_grad=True)
        p  = torch.zeros((v.shape[0], self.NT, v.shape[1], v.shape[2]), device=self.device, requires_grad=True)
        
        for i in range(self.nt):
            p = c2*(torch.roll(p1, 1, dims = -2) + torch.roll(p1, -1, dims = -2) + torch.roll(p1, 1, dims = -1)+ torch.roll(p1, -1, dims = -1)) +c3*(torch.roll(p1, 2, dims = -2) + torch.roll(p1, -2, dims = -2) + torch.roll(p1, 2, dims = -1)+ torch.roll(p1, -2, dims = -1))
            p = torch.einsum('ijk,iljk->iljk', alpha,p)#p*alpha[:,None,:,:]
            p += torch.einsum('ijk,iljk->iljk', temp1,p1) - torch.einsum('ijk,iljk->iljk', temp0,p0)#p1*temp1[:,None,:,:] + p0*temp0[:,None,:,:]
            p = p.reshape((v.shape[0],self.NT, 1, v.shape[1]*v.shape[2]))
            for j in range(self.NT):
                p[:,j,0,self.in_inds[j]] += beta[:,j]*self.source[i]
            meas.append(p[:,:,:,self.meas_inds])
            p = p.reshape((v.shape[0],self.NT, v.shape[1],v.shape[2]))
            p0=p1
            p1=p
        
        return torch.cat(meas, dim=2)



   


