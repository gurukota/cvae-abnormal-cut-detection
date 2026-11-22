"""Physically bounded feature coordinates, categorical twin, RF/XGB/LSTM inputs."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
PARAMS=['N','ap','ae','F','Z','D']
MATERIALS=['AA 6061','AISI 1018','AISI 4140']
COORD_DIM=72
RAW_DIM=82
EXTRA_SCALE=np.array([1000.]+[1000.,1000.,1.,1.,.001]*4+[.001],dtype=float)
NYQUIST=np.array([24000,24000,24000,25600,25600],float)
torch.set_num_threads(2)

def logit(x):
    x=np.clip(x,1e-5,1-1e-5);return np.log(x/(1-x))

def primitive(raw):
    """72 coordinates: 50 bounded waveform descriptors plus 22 nonnegative cut-level MTConnect summaries."""
    a=np.asarray(raw[...,:60],dtype=float).reshape(*raw.shape[:-1],5,12)
    rms=np.maximum(a[...,0],1e-8);crest=np.maximum(a[...,1]/rms,1.00001)
    skew=a[...,4]
    coords=[np.log(rms),np.log(crest-1),np.arcsinh(skew),
            np.log(np.maximum(a[...,3]+2-skew**2,1e-5)),
            logit(a[...,5]/rms),logit((a[...,6]/rms-1/crest)/(1-1/crest)),
            logit(a[...,7]/NYQUIST),logit(a[...,8]/NYQUIST),
            np.log(np.maximum(a[...,9],1e-8)/np.maximum(a[...,11],1e-8)),
            np.log(np.maximum(a[...,10],1e-8)/np.maximum(a[...,11],1e-8))]
    base=np.stack(coords,-1).reshape(*raw.shape[:-1],50)
    extra=np.log1p(np.maximum(raw[...,60:],0)/EXTRA_SCALE)
    return np.concatenate([base,extra],axis=-1).astype('float32')

def physical(coords):
    q=np.asarray(coords[...,:50],dtype=float).reshape(*coords.shape[:-1],5,10)
    sig=lambda x:1/(1+np.exp(-np.clip(x,-30,30)))
    exp=lambda x:np.exp(np.clip(x,-20,20))
    rms=exp(q[...,0]);crest=1+exp(q[...,1]);peak=rms*crest
    skew=np.sinh(np.clip(q[...,2],-10,10));kurt=skew**2-2+exp(q[...,3])
    std=rms*sig(q[...,4]);ma=rms*(1/crest+(1-1/crest)*sig(q[...,5]))
    cent=NYQUIST*sig(q[...,6]);dom=NYQUIST*sig(q[...,7])
    logits=np.stack([q[...,8],q[...,9],np.zeros_like(rms)],-1)
    bands=np.exp(logits-logits.max(-1,keepdims=True));bands/=bands.sum(-1,keepdims=True)
    vals=[rms,peak,crest,kurt,skew,std,ma,cent,dom,bands[...,0],bands[...,1],bands[...,2]]
    base=np.stack(vals,-1).reshape(*coords.shape[:-1],60)
    extra=np.expm1(np.clip(coords[...,50:],0,25))*EXTRA_SCALE
    return np.concatenate([base,extra],axis=-1).astype('float32')

def constraint_counts(raw):
    a=np.asarray(raw[...,:60],dtype=float).reshape(-1,5,12);eps=1e-4
    rms,peak,crest,kurt,skew,std,ma=[a[...,i] for i in range(7)]
    return {
        'nonfinite':int((~np.isfinite(raw)).sum()),
        'negative_mtc_measurement':int((raw[...,60:]<0).sum()),
        'negative_amplitude':int((a[:,:,[0,1,5,6]]<0).sum()),
        'peak_below_rms':int((peak+eps<rms).sum()),
        'std_above_rms':int((std>rms+eps*(1+rms)).sum()),
        'mean_abs_above_rms':int((ma>rms+eps*(1+rms)).sum()),
        'mean_abs_lower_bound':int((ma*peak+eps*(1+rms*rms)<rms*rms).sum()),
        'crest_identity':int((np.abs(crest*rms-peak)>eps*(1+peak)).sum()),
        'moment_bound':int((kurt+2+eps*(1+skew**2)<skew**2).sum()),
        'frequency_bounds':int(((a[:,:,7:9]<0)|(a[:,:,7:9]>NYQUIST[None,:,None]+eps)).sum()),
        'band_bounds':int(((a[:,:,9:]<0)|(a[:,:,9:]>1)).sum()),
        'band_sum':int((np.abs(a[:,:,9:].sum(-1)-1)>eps).sum())}

def parameter_matrix(frame):
    # Tool ID, its asterisk, original label and both targets are excluded.
    params=np.log1p(frame[PARAMS].to_numpy(float))
    mats=np.stack([(frame.Workpiece==v).to_numpy() for v in MATERIALS],1).astype(float)
    direction=(frame.direction.astype(str).str.upper()=='UP').to_numpy(float)[:,None]
    return np.c_[params,mats,direction].astype('float32')

class Transform:
    def __init__(self,raw,frame):
        z=primitive(raw);flat=z.reshape(-1,COORD_DIM)
        self.mean=flat.mean(0);self.std=np.maximum(flat.std(0),.1)
        self.low=flat.min(0)-.25*np.maximum(flat.std(0),.1)
        self.high=flat.max(0)+.25*np.maximum(flat.std(0),.1)
        p=parameter_matrix(frame);self.pm=p.mean(0);self.ps=np.maximum(p.std(0),.1)
    def x(self,raw):return ((primitive(raw)-self.mean)/self.std).astype('float32')
    def inverse(self,x):
        coords=np.clip(np.asarray(x)*self.std+self.mean,self.low,self.high)
        # Cut-level accelerometer and low-rate MTConnect summaries remain constant across the eight audio windows.
        if coords.ndim==3:coords[:,:,30:]=coords[:,:,30:].mean(1,keepdims=True)
        return physical(coords)
    def p(self,frame):return ((parameter_matrix(frame)-self.pm)/self.ps).astype('float32')
    def condition(self,frame,target_tool=None):
        a=frame.abnormal_cut.to_numpy(int)
        d=frame.defective_tool.to_numpy(int) if target_tool is None else np.full(len(frame),target_tool)
        return np.c_[np.eye(2)[a],np.eye(2)[d],self.p(frame)].astype('float32')
    def detector(self,raw,frame):
        x=self.x(raw);p=self.p(frame)
        # No condition labels are included. True temporal input for LSTM.
        return np.concatenate([x,np.repeat(p[:,None,:],x.shape[1],axis=1)],axis=2).astype('float32')

def tabular(sequence):
    return np.concatenate([sequence[:,:,:COORD_DIM].mean(1),sequence[:,:,:COORD_DIM].std(1),
                           sequence[:,:,:COORD_DIM].max(1),sequence[:,0,COORD_DIM:]],1)

class Twin(nn.Module):
    def __init__(self,cond_dim,latent=12,hidden=96,beta=.01):
        super().__init__();self.latent=latent
        self.encoder=nn.Sequential(nn.Linear(COORD_DIM+cond_dim,hidden),nn.ELU(),nn.Linear(hidden,hidden),nn.ELU())
        self.mu=nn.Linear(hidden,latent);self.lv=nn.Linear(hidden,latent)
        self.decoder=nn.Sequential(nn.Linear(latent+cond_dim,hidden),nn.ELU(),nn.Linear(hidden,hidden),nn.ELU(),nn.Linear(hidden,COORD_DIM))
    def encode(self,x,c):
        h=self.encoder(torch.cat([x,c],-1));return self.mu(h),self.lv(h).clamp(-10,8)
    def decode(self,z,c):return self.decoder(torch.cat([z,c],-1))
    def forward(self,x,c,random=True):
        mu,lv=self.encode(x,c);z=mu+torch.randn_like(mu)*torch.exp(.5*lv) if random else mu
        return self.decode(z,c),mu,lv

class DetectorLSTM(nn.Module):
    def __init__(self,input_dim,hidden=32):
        super().__init__();self.rnn=nn.LSTM(input_dim,hidden,batch_first=True)
        self.head=nn.Sequential(nn.Linear(hidden,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self,x):
        _,(h,_)=self.rnn(x);return self.head(h[-1]).squeeze(-1)
