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


@torch.no_grad()
def evaluation(*,model,device,num_classes,validation_dataloader):
    model.eval()
    loss_sum=0
    size_sum=0
    for inputids,labels  in validation_dataloader:
        inputids = inputids.to(device)
        labels = labels.to(device)
        step_size=labels.shape[0]
        labels = torch.reshape(labels, (-1,))
        logits = model(inputids)  # (batch_size,seq_len,num_items+1)
        logits = torch.reshape(logits, (-1, logits.shape[-1]))
        # ignore_index,will ignore the value of label is 0
        loss = F.cross_entropy(input=logits, target=labels)
        loss_sum+=loss.cpu().item()*step_size
        size_sum+=step_size
    validation_loss=np.round(loss_sum/size_sum,4)
    normal_loss=0.0
    if num_classes is not None and num_classes>0:
        normal_loss = np.round(validation_loss/np.log(num_classes),4)
    logger.info({
        "stage":"validation",
        "validation_number":size_sum,
        "validation_loss":validation_loss,
        "normal_loss":normal_loss,
        "validation_ppl":np.round(np.exp(validation_loss),4),
        "num_classes":num_classes
    })
    model.train()
    return validation_loss
evaluation(model=model,
           num_classes=16000,
           device="cpu",
           validation_dataloader=validation_dataloader)










