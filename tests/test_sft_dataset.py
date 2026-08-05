from llmbrew.dataset import SFTDataset
from llmbrew.config import data_config
from torch.utils.data import DataLoader
import os
context_length=512
batch_size=2
data_path=data_config.llmbrew_sft_train_path
dataset=SFTDataset(data_path=data_path,
             context_length=context_length,batch_size=batch_size)
for feature,label in dataset:
    print(feature)
    print(label)
    print(feature.tolist())
    print(label.tolist())
    print(type(feature))
    print(type(label))
    print(feature.shape)
    print(label.shape)
    break;
dataloader=DataLoader(dataset,batch_size=None)
for feature,label in dataloader:
    print(feature)
    print(label)
    print(feature.tolist())
    print(label.tolist())
    print(type(feature))
    print(type(label))
    print(feature.shape)
    print(label.shape)
    break;
