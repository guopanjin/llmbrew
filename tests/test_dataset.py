from llmbrew.dataset import PretrainDataset
from llmbrew.config import data_config
from torch.utils.data import DataLoader
import os
context_length=512
batch_size=2
data_path=data_config.validation_bin_path
dataset=PretrainDataset(data_path=data_path,
             context_length=context_length,batch_size=batch_size)
for feature,label in dataset:
    print(feature)
    print(label)
    print(type(feature))
    print(type(label))
    print(feature.shape)
    print(label.shape)
    break;
dataloader=DataLoader(dataset,batch_size=None)
