import torch
import torch.nn as nn
import torch.nn.functional as F
from llmbrew.utils import Logger
logger=Logger.get_logger()
class LLMBrewConfig():
    def __int__(self,
                vocab_size:int=16000,
                hidden_dim:int=320,
                num_heads:int=5,
                intermediate_dim:int=864,
                num_decoder_layers:int=4
                ):
        self.vocab_size=vocab_size
        self.hidden_dim=hidden_dim
        self.num_heads=num_heads
        self.intermediate_dim=intermediate_dim

class LLMBrewModel(nn.Module):
    def __init__(self,llmbrewconfig:LLMBrewConfig):
        super(LLMBrewModel, self).__init__()
        self.llmbrewconfig=llmbrewconfig
    '''
    input:
    x:(feature_tensor,label_tensor)
    x_shape:([batch_size,seq_len],[batch_size,seq_len])
    '''
    def forward(self,x):
        pass
