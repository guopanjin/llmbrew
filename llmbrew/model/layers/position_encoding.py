import torch
import torch.nn as nn
import torch.nn.functional as F

class ROPE(nn.Module):
    def __init__(self,
                rope_theta:int=10000
                ):
        super().__init__()
        self.rope_theta=rope_theta

    '''
    input:(batch_size,num_heads,seq_len,head_dim)
    rotate:(x1,x2)==>(x1*cos-x2*sin, x1*sin+x2*cos)
    angle:position_index*wi 
    wi=rope_theta^-(2i/d) i->0-->d/2-i
    '''
    def forward(self,x):
        if len(x.shape)!=4:
            raise ValueError(f"inputs dimention must be,but we got {len(x.shape)}")
        batch_size,num_heads,seq_len,head_dim=x.shape
        if head_dim<1 or head_dim%2!=0:
            raise ValueError(f"head_dim must be even number")
        #setup angle matrix
        position_index=torch.arange(seq_len,dtype=torch.float32, device=x.device)#(seq_len,)
        w_index=torch.arange(head_dim/2,dtype=torch.float32,device=x.device)#(hidden_dim/2,),the arange will not include the last value head_dim/2
        w_i=torch.pow(self.rope_theta,-2*w_index/head_dim)
        angles=torch.unsqueeze(position_index,dim=-1) @ torch.unsqueeze(w_i,dim=0)#(seq_len,hidden_dim/2)
        angles=torch.repeat_interleave(angles,2,dim=-1) #(seq_len,hidden_dim)
        cos_theta=torch.cos(angles)
        sin_theta=torch.sin(angles)
        #calculate the postion score
        x1=torch.reshape(x,(batch_size,num_heads,seq_len,-1,2))
        x1=torch.flip(x1,dims=(-1,)).reshape((batch_size,num_heads,seq_len,head_dim))
        sign=torch.where(torch.arange(head_dim,device=x.device)%2==0,-1,1)#(head_dim,)
        sign=torch.unsqueeze(sign,dim=0) #(1,head_dim)
        rotaty_x=x1*sign #(batch_size,num_heads,seq_len,head_dim)
        return cos_theta*x+sin_theta*rotaty_x #(batch_size,num_heads,seq_len,head_dim)