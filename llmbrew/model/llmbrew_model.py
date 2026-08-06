import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from llmbrew.utils import Logger
from llmbrew.model.layers import DecoderLayer
from llmbrew.model.layers import RMSNormal
from llmbrew.model.layers.initialization_functions import init_embedding
logger=Logger.get_logger()
class LLMBrewConfig():
    def __init__(self,
                vocab_size:int=16000,
                hidden_dim:int=320,
                num_heads:int=5,
                intermediate_dim:int=864,
                num_decoder_layers:int=4,
                causal_mask:int=True
                ):
        self.vocab_size=vocab_size
        self.hidden_dim=hidden_dim
        self.num_heads=num_heads
        self.intermediate_dim=intermediate_dim
        self.num_decoder_layers=num_decoder_layers
        self.causal_mask=causal_mask

class LLMBrewModel(nn.Module):
    def __init__(self,llmbrewconfig:LLMBrewConfig):
        super(LLMBrewModel, self).__init__()
        self.llmbrewconfig=llmbrewconfig
        self.embedding_table = nn.Embedding(num_embeddings=self.llmbrewconfig.vocab_size, embedding_dim=self.llmbrewconfig.hidden_dim)
        decoder_layer_list=[DecoderLayer(hidden_dim=self.llmbrewconfig.hidden_dim,
                                          num_heads=self.llmbrewconfig.num_heads,
                                          intermediate_dim=self.llmbrewconfig.intermediate_dim,
                                          causal_mask=self.llmbrewconfig.causal_mask) for _ in range(self.llmbrewconfig.num_decoder_layers)]
        self.decoder_layers=nn.ModuleList(decoder_layer_list)
        self.final_normal=RMSNormal(self.llmbrewconfig.hidden_dim)
        logger.info(f"llmbrewmodel config {self.llmbrewconfig.__dict__}")
        self.apply(lambda m: init_embedding(m))
    '''
    input:
    input_ids:[batch_size,seq_len],
    output:logits #[batch_size,seq_len,vocab_size]
    '''
    def forward(self,inputs_id):
        hidden_state=self.embedding_table(inputs_id) #(batch_size,seq_len,hidden_dim)
        for decoder_layer in self.decoder_layers:
            hidden_state=decoder_layer(hidden_state)#(batch_size,seq_len,hidden_dim)
        hidden_state=self.final_normal(hidden_state)
        logits=hidden_state@self.embedding_table.weight.T #(batch_size,seq_len,vocab_size)
        return logits
    '''
    input:
    input_ids:[batch_size,prompt_len]
    TODO:need to change code to support batch
    '''
    @torch.no_grad()
    def generate(self,input_ids,
                 temperature=1.0,
                 top_k=40,
                 do_sample=True,
                 eos_token_id=2,
                 max_new_tokens=50):
        prompt_len=input_ids.shape[1]
        #validate argements
        temperature=np.clip(temperature,0.1,100)
        for _ in range(max_new_tokens):
            output=self.forward(input_ids)[:,-1,:] #[batch_size,vocab_size]
            if do_sample:
                output = output / temperature
                output,indices=torch.sort(output,dim=-1,descending=True)
                indices=indices[:,:top_k]
                output=output[:,:top_k]
                output=F.softmax(output,dim=-1)
                sample_index=torch.multinomial(output,num_samples=1)
                last_output=indices[:,sample_index].reshape(shape=(-1,1))
            else:
                last_output=torch.argmax(output,dim=-1,keepdim=True)#[batch_size,1]
            new_inputs=torch.concat([input_ids,last_output],dim=-1)#[batch_size,seq_len+1]
            input_ids=new_inputs
            if last_output[:, 0] == eos_token_id:
                break;
        return input_ids[:,prompt_len:]

    @torch.no_grad()
    def generate_stream(self, input_ids,
                 temperature=1.0,
                 top_k=40,
                 do_sample=True,
                 eos_token_id=2,
                 max_new_tokens=50):
        # validate argements
        temperature = np.clip(temperature, 0.1, 100)
        for _ in range(max_new_tokens):
            output = self.forward(input_ids)[:, -1, :]  # [batch_size,vocab_size]
            if do_sample:
                output = output / temperature
                output, indices = torch.sort(output, dim=-1, descending=True)
                indices = indices[:, :top_k]
                output = output[:, :top_k]
                output = F.softmax(output, dim=-1)
                sample_index = torch.multinomial(output, num_samples=1)
                last_output = indices[:, sample_index].reshape(shape=(-1, 1))
            else:
                last_output = torch.argmax(output, dim=-1, keepdim=True)  # [batch_size,1]
            new_inputs = torch.concat([input_ids, last_output], dim=-1)  # [batch_size,seq_len+1]
            input_ids = new_inputs
            new_token_id=last_output[:, 0].item()
            if  new_token_id== eos_token_id:
                break;
            yield new_token_id




