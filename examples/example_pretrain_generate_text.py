import torch
import torch.nn.functional as F
import os
from llmbrew.config import data_config,model_artifacts_config
from llmbrew.model import LLMBrewModel,LLMBrewConfig
from llmbrew.trainer import PretrainTrainer
from llmbrew.utils import get_device,prevent_sleep
from llmbrew.dataset import PretrainDataset
from torch.utils.data import Dataset,DataLoader
import numpy as np
import os
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from llmbrew.config import data_config
from llmbrew.utils import  prevent_sleep
from transformers import PreTrainedTokenizerFast,AutoTokenizer
from llmbrew.config import data_config

from llmbrew.utils.log_utils import Logger
logger=Logger.get_logger()
'''
final_model_state={"model_state_dict":self.model.state_dict(),
                           "config":self.model.llmbrewconfig,
                           "num_tokens":self.global_tokens
                           }
'''
validation_data_path=data_config.validation_bin_path
context_length=512
validation_batch_size=100
validation_dataset = PretrainDataset(
    data_path=validation_data_path,
    context_length=context_length,
    batch_size=validation_batch_size,
    is_train=False
)
validation_dataloader = DataLoader(validation_dataset, batch_size=None)

model_file_path=model_artifacts_config.model_outputs_pretrain_final_model_path+"pretrain_model.pt"
model_states=torch.load(os.path.expanduser(model_file_path),weights_only=False)
model_config=model_states["config"].__dict__
num_tokens=model_states["num_tokens"]
model_state_dict=model_states["model_state_dict"]
print("num_tokens",num_tokens)
print(model_config)

llmbrewconfig= LLMBrewConfig(**model_config)
model=LLMBrewModel(llmbrewconfig=llmbrewconfig)
model.load_state_dict(state_dict=model_state_dict)
model=model.to("cpu")


tokenizer_path="../artifacts/tokenizer/llmbrew_tokenizer_v1"
hf_tokenizer:PreTrainedTokenizerFast=AutoTokenizer.from_pretrained(tokenizer_path,use_fast=True)

BASELINE_PROMPTS = [
    # 事实续写
    "中国的首都是",
    "The capital of France is",

    # 常识 / 推理
    "如果明天下雨，我们就",
    "Water boils at a temperature of",

    # 指令式 —— SFT 前后差异最明显的一类
    "请解释什么是机器学习。",
    "Explain what machine learning is.",

    # 代码
    "def fibonacci(n):",

    # 开放续写
    "从前有一座山，山里",
    "Once upon a time, there was a",

    # 无条件生成 —— 反映语料整体分布
    "",
]

def token_encoder(text):
    tokenids = hf_tokenizer.encode(text, add_special_tokens=False)
    return tokenids

def token_decoder(tokenids):
    decoded = hf_tokenizer.decode(tokenids,
                                  skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False
                                  )
    return decoded

def test_decoder_encoder():
    tokenids = token_encoder("什么是深度学习?")
    print(tokenids)
    text = token_decoder(tokenids)
    print(text)
    pass

@torch.no_grad()
def generate_text(input_text,
                  eos_token_id=2,
                  max_new_tokens=50):
    model.eval()
    tokenids=token_encoder(input_text)
    if not tokenids:# in case null value
        tokenids.append(eos_token_id)
    tokenids =torch.tensor(tokenids,dtype=torch.long)
    '''
    input:[batch_size,inputids] 
    return:[batch_size,max_new_tokens]
    '''
    tokenids=torch.unsqueeze(tokenids,dim=0)

    output_ids=model.generate(input_ids=tokenids,
                   max_new_tokens=max_new_tokens,
                   eos_token_id=2
                   )
    output_tokens=output_ids.detach().tolist()[0]
    response=token_decoder(output_tokens)
    return response

if __name__ == '__main__':
    for prompt in BASELINE_PROMPTS:
        response=generate_text(prompt,max_new_tokens=50)
        print(f"prompt:{prompt}")
        print(f"response:{response}")






