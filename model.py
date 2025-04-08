import torch
import torch.nn as nn
import math
import numpy as np

class TextEncoding(nn.Module):
    def __init__(self,dmodel,vocab_size):
        super().__init__()
        self.dmodel=dmodel
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(self.vocab_size,self.dmodel)

    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.dmodel)


class PositionalEmbedding(nn.Module):
    def __init__(self,dmodel,seq_length,dropout):
        super().__init__()
        self.dmodel=dmodel
        self.seq_length=seq_length
        self.dropout=nn.Dropout(dropout)
        # first lets make a matrix of dimension(seq_length,dmodel) 
        inp=torch.zeros(self.seq_length,self.dmodel)

        '''now lets apply the formulas for pe:
        for even we have sin(pos/10000^(2i/dmodel))  
        for odd we have cos(pos/10000^(2i/dmodel))'''

        position=torch.arange(0,self.seq_length,1).unsqueeze(1) #[seq_length,1]
        i=torch.arange(0,dmodel,2).unsqueeze(0)#[1,dmodel]
        denominator=10000**((2*i)/self.dmodel)
        inp[:,0::2]=torch.sin(position/denominator)
        inp[:,1::2]=torch.cos(position/denominator)
        inp=inp.unsqueeze(1)#(1,seq_len,dmodel)-> for batches
        #-> to save a tensor(not learned) to module: add to buffer
        self.register_buffer("inp",inp)

    def forward(self,x):
        x=x+self.inp[:,:x.shape[1]:].requires_grad(False)
            

class LayerNorm(nn.Module):
    def __init__(self,eps=10**-6):
        super().__init()
        self.eps=eps
        self.alpha=nn.Parameter(nn.ones(1))
        self.bias=nn.Parameter(nn.zeros(1))
    
    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        deviation=x.std(dim=-1,keepdim=True)
        return self.alpha*(x-mean)/math.sqrt(deviation+self.eps)+self.bias

#this is to connect the layers  
class FeedForward(nn.Module):
    def __init__(self,dmodel,d_ff,dropout):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(dmodel,d_ff),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(d_ff,dmodel)
        )
    def forward(self,x):
        return self.model(x)

#now lets move on to the multi head attention
class MHA(nn.Module):
    def __init__(self,dmodel,h,dropout):
        self.h=h
        self.dmodel=dmodel
        if(dmodel%h!=0){
            print("d_model is not divisible by h")
        }
        self.d_k=self.model//h
        self.d_v=self.model//h
        self.d_q=self.model//h
        self.w_q=nn.Linear(dmodel,dmodel)
        self.w_k=nn.Linear(dmodel,dmodel)
        self.w_v=nn.Linear(dmodel,dmodel)
        self.w_o=nn.Linear(h*d_v,dmodel)

