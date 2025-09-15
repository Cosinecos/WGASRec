import math, torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import SWTForward
from ..utils.registry import register
class TFLN(nn.Module):
    def __init__(self,d,eps=1e-12):
        super().__init__()
        self.g=nn.Parameter(torch.ones(d))
        self.b=nn.Parameter(torch.zeros(d))
        self.eps=eps
    def forward(self,x):
        m=x.mean(-1,keepdim=True)
        v=(x-m).pow(2).mean(-1,keepdim=True)
        return (x-m)/torch.sqrt(v+self.eps)*self.g+self.b
class FFN(nn.Module):
    def __init__(self,d,p=0.2):
        super().__init__()
        self.fc1=nn.Linear(d,4*d)
        self.fc2=nn.Linear(4*d,d)
        self.do=nn.Dropout(p)
        self.ln=TFLN(d)
    def forward(self,x):
        y=self.fc1(x)
        y=y*0.5*(1.0+torch.erf(y/math.sqrt(2.0)))
        y=self.fc2(y)
        return self.ln(y+x)
class AddAttn(nn.Module):
    def __init__(self,d,p=0.1):
        super().__init__()
        self.W=nn.Linear(d,d)
        self.q=nn.Parameter(torch.randn(d)*0.02)
        self.do=nn.Dropout(p)
    def forward(self,x,mask):
        s=torch.tanh(self.W(x))@self.q
        s=s.masked_fill(mask<0.5,-1e4)
        a=F.softmax(s,dim=-1)
        y=(a.unsqueeze(-1)*x).sum(1)
        return y
class Boundary(nn.Module):
    def __init__(self,d,L=4):
        super().__init__()
        self.L=L
        self.l=nn.Parameter(torch.randn(1,L,d)*0.02)
        self.r=nn.Parameter(torch.randn(1,L,d)*0.02)
    def forward(self,x,tau):
        if tau<=0: return x
        b=x.size(0)
        return torch.cat([self.l[:,:tau].expand(b,-1,-1),x,self.r[:,:tau].expand(b,-1,-1)],1)
    def crop(self,x,tau,T):
        if tau<=0: return x[:,:T]
        return x[:,tau:tau+T]
def band_energy(x):
    return x.pow(2).mean(dim=(1,2))
def band_sfm(x,eps=1e-8):
    p=x.pow(2)+eps
    geo=torch.exp(torch.log(p).mean(dim=(1,2)))
    ari=p.mean(dim=(1,2))
    return geo/(ari+eps)
class SubbandUnit(nn.Module):
    def __init__(self,d,drop=0.2):
        super().__init__()
        self.attn=AddAttn(d,drop*0.5)
        self.ffn=FFN(d,drop)
class WGASBlock(nn.Module):
    def __init__(self,d,J=2,drop=0.2,gate_tau=1.0):
        super().__init__()
        self.swt=SWTForward(J=J,wave='sym4',start_level=0)
        self.sub=nn.ModuleList([SubbandUnit(d,drop) for _ in range(J+1)])
        self.gin=2+J+1
        self.gmlp=nn.Sequential(nn.Linear(self.gin, max(8,4*(J+1))), nn.GELU(), nn.Linear(max(8,4*(J+1)),1))
        self.ln=TFLN(d)
        self.drop=nn.Dropout(drop)
        self.gate_tau=gate_tau
    def forward(self,x,mask):
        b,t,c=x.size()
        y=x
        ca,cd=self.swt(x.transpose(1,2))
        bands=[ca.transpose(1,2)]+[h.transpose(1,2) for h in cd]
        stats=[]
        pooled=[]
        for i,bn in enumerate(bands):
            pooled.append(self.sub[i].attn(bn,mask))
            e=band_energy(bn).view(-1,1)
            s=band_sfm(bn).view(-1,1)
            lv=torch.full_like(e, float(i))
            stats.append(torch.cat([e,s,lv],-1))
        S=torch.stack(stats,1)
        g=F.softmax(g/self.gate_tau,dim=-1)
        H=torch.stack(pooled,1)
        z=(g.unsqueeze(-1)*H).sum(1)
        z=z.unsqueeze(1).expand(-1,t,-1)
        z=self.drop(z)
        z=self.ln(z+y)
        return z,g
class ChebLayer(nn.Module):
    def __init__(self,d,K=2):
        super().__init__()
        self.K=K
        self.theta=nn.ParameterList([nn.Parameter(torch.randn(d,d)*0.02) for _ in range(K+1)])
    def forward(self,Lt,X):
        T0=X
        out=T0@self.theta[0]
        if self.K>=1:
            T1=Lt@X
            out=out+T1@self.theta[1]
        for k in range(2,self.K+1):
            Tk=2*Lt@T1 - T0
            out=out+Tk@self.theta[k]
            T0,T1=T1,Tk
        return out
class GraphStack(nn.Module):
    def __init__(self,d,Lg=2,K=2):
        super().__init__()
        self.layers=nn.ModuleList([ChebLayer(d,K) for _ in range(Lg)])
    def forward(self,Lt,E):
        H=E
        for l in self.layers:
            H=l(Lt,H)
        return H
class WGASRec(nn.Module):
    def __init__(self,num_items,hidden_size=128,max_seq_length=100,num_layers=2,dropout=0.2,J=2,gate_tau=1.0,Lg=2,K=2,Ltilde=None):
        super().__init__()
        self.ni=num_items
        self.d=hidden_size
        self.T=max_seq_length
        self.emb=nn.Embedding(num_items+1,hidden_size,padding_idx=0)
        self.pos=nn.Embedding(max_seq_length+64,hidden_size)
        self.bd=Boundary(hidden_size, L=8)
        self.blocks=nn.ModuleList([WGASBlock(hidden_size,J,dropout,gate_tau) for _ in range(num_layers)])
        self.gnn=GraphStack(hidden_size,Lg,K)
        self.drop=nn.Dropout(dropout)
        self.bias=nn.Parameter(torch.zeros(num_items+1))
        self.register_buffer("Ltilde", Ltilde if Ltilde is not None else torch.zeros(num_items+1,num_items+1))
    def encode(self,seq,mask,tau):
        b,t=seq.size()
        pos=torch.arange(t,device=seq.device).unsqueeze(0).expand(b,-1)
        x=self.emb(seq)+self.pos(pos)
        x=self.bd(x,int(tau.max().item()))
        m=torch.cat([torch.zeros(b,int(tau.max().item()),device=mask.device),mask],1)
        for blk in self.blocks:
            x,_=blk(x,m)
        x=self.bd.crop(x,int(tau.max().item()),t)
        x=self.ln(x)
        return x
    def item_bank(self):
        E=self.emb.weight
        H=self.gnn(self.Ltilde,E)
        return H
    def forward(self,seq,mask,tau):
        enc=self.encode(seq,mask,tau)
        L=mask.long().sum(1)
        idx=(L-1).clamp(min=0).view(-1,1,1).expand(-1,1,self.d)
        q=torch.gather(enc,1,idx).squeeze(1)
        H=self.item_bank()
        logits=q@H.t()+self.bias
        return logits
@register('model','WGASRec')
def build(**cfg):
    return WGASRec(**cfg)
