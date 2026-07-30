import torch
import torch.nn as nn
import torch.nn.functional as F
from llmbrew.model.layers import ROPE
'''
causal self-attention only for decoder layer of transformer
'''
class MultiHeadAttention(nn.Module):
    def __init__(self,
                 hidden_dim:int,
                 num_heads:int,
                 causal_mask:bool=True
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.causal_mask=causal_mask
        if hidden_dim%num_heads!=0:
            raise ValueError(f"hidden_dim must be divisable by num_heads")
        self.head_dim=hidden_dim//num_heads
        self.q_layer=nn.Linear(self.hidden_dim,self.hidden_dim)
        self.k_layer=nn.Linear(self.hidden_dim,self.hidden_dim)
        self.v_layer=nn.Linear(self.hidden_dim,self.hidden_dim)
        self.out_proj_layer=nn.Linear(self.hidden_dim,self.hidden_dim)
        self.rope=ROPE()

    '''
    x:(batch_size,seq_len,hidden_dim)
    '''
    def forward(self,x):
        batch_size,seq_len,_=x.shape
        q=self.q_layer(x)
        k=self.k_layer(x)
        v=self.v_layer(x)
        #split heads
        q=torch.reshape(q,(batch_size,seq_len,self.num_heads,self.head_dim))
        q=torch.transpose(q,1,2) #(batch_size,num_heads,seq_len,head_dim)
        k = torch.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = torch.transpose(k, 1, 2)  # (batch_size,num_heads,seq_len,head_dim)
        v = torch.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = torch.transpose(v, 1, 2)  # (batch_size,num_heads,seq_len,head_dim)
        #add rope
        q=self.rope(q)
        k=self.rope(k)
        attention_score=q@k.T #(batch_size,num_heads,seq_len,seq_len)
        if self.causal_mask:
            mask=torch.ones(size=(seq_len,seq_len),device=x.device)
            mask=torch.triu(mask,diagonal=1).to(torch.bool)
            attention_score=torch.masked_fill(attention_score,mask,torch.finfo(x.dtype).min)
        attention_weights=F.softmax(attention_score,dim=-1)
        attention_output=attention_weights@v #(batch_size,num_heads,seq_len,head_dim)
        attention_output=torch.transpose(attention_output,1,2).reshape((batch_size,seq_len,self.hidden_dim))
        output=self.out_proj_layer(attention_output)
        return output