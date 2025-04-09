import torch
import torch.nn as nn
import torch.nn.Functional as F
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
        super().__init__()
        self.h=h
        self.dmodel=dmodel
        if(dmodel%h!=0):
            print("d_model is not divisible by h")
        
        self.d_k=self.dmodel//h
        self.d_v=self.dmodel//h
        self.d_q=self.dmodel//h
        self.w_q=nn.Linear(dmodel,dmodel)
        self.w_k=nn.Linear(dmodel,dmodel)
        self.w_v=nn.Linear(dmodel,dmodel)
        self.w_o=nn.Linear(h*d_v,dmodel)
        self.dropout=nn.Dropout(dropout)
    
    def Attention(self,query, key, value, mask=None):
        """
        Args:
            query: (batch_size, num_heads, seq_len_q, d_k)
            key: (batch_size, num_heads, seq_len_k, d_k)
            value: (batch_size, num_heads, seq_len_v, d_v)
            mask: Optional mask to prevent attention to certain positions
        """
        self.d_k=query.size(-1)
        self.score=torch.matmul(self.query,self.key.transpose(-2,-1)/math.sqrt(self.d_k))

        
        if mask is not None:
            scores=scores.masked_fill(mask==0,float('-inf'))

        attention_weights=F.softmax(scores,dim=-1)
        return torch.matmul(attention_weights,self_value)


    def forward(self,x,query,key,value,mask):
        self.query=self.w_q(query)#old dim: (batch,seq_len,dmodel)-> new dim: (batch,seq_len,dmodel) preserves 
        self.key=self.w_k(key)
        self.value=self.w_v(value)
        #the linear projections 
        query=self.w_q(query)#dim->(batch,seq_len,dmodel)
        key=self.w_k(key)
        value=self.w_v(value)
        #splitting into heads by dmodel
        Q=query.view(torch.view(query.shape[0],query.shape[1],self.h,self.dmodel//self.h))
        #dimension->(batch,seq_len,num_heads,head_dim)
        #eg->earlier dim->(32,20,512) new dim->(32,20,8,64)
        K=key.view(torch.view(query.shape[0],query.shape[1],self.h,self.dmodel//self.h)).transpose(1,2)
        V=value.view(torch.view(query.shape[0],query.shape[1],self.h,self.dmodel//self.h)).transpose(1,2)
        
        out=self.Attention(Q,K,V,mask)
        #now joining the heads
        out=out.transpose(1,2).contiguous.view(batch,seq_len,self.dmodel)
        return self.w_o(out)





