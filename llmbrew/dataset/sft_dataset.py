import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import IterableDataset,get_worker_info
from llmbrew.utils import Logger
import json
'''
Build the streamingDataset based on the pytorch API
return (features,labels)
TODO:dynamically pad data
'''
logger=Logger.get_logger()
class  SFTDataset(IterableDataset):
   def __init__(self,
                data_path,
                padding_id:int=0,
                label_padding_id:int=-100,
                context_length=512,
                batch_size=1000
                ):
       self.data_path=os.path.expanduser(data_path)
       self.context_length=context_length
       self.batch_size=batch_size
       self.padding_id=padding_id
       self.label_padding_id=label_padding_id
       self.file_list=sorted(self.__get_file_list())# make sure the dataset is stable
   def __get_file_list(self):
       file_list=[]
       if os.path.isdir(self.data_path):
           for root, dirs, files in os.walk(self.data_path):
               for file in files:
                   file_list.append(os.path.expanduser(os.path.join(root, file)))
       else:
           file_list.append(os.path.expanduser(self.data_path))
       logger.info(f"file_list:{file_list}")
       return file_list
   '''
   return (features,labels)==>((batch_size,context_length),(batch_size,context_length) )
   will shift in the loss function.
   '''
   def __parse_data(self, file):
       with open(file,"r") as f:
           batch_cnt = 0
           batch_input_ids=[]
           batch_label_ids=[]
           for json_str in f:
               json_data=json.loads(json_str)
               input_ids=json_data["input_ids"]
               label_ids=input_ids.copy()
               padding_len=self.context_length-len(input_ids)
               final_input_ids=input_ids+padding_len*[self.padding_id]
               prompt_len=int(json_data["prompt_len"])
               final_label_ids=prompt_len*[self.label_padding_id]+label_ids[prompt_len:]+padding_len*[self.label_padding_id]
               batch_cnt+=1
               batch_input_ids.append(final_input_ids)
               batch_label_ids.append(final_label_ids)
               if batch_cnt>1 and batch_cnt%self.batch_size==0:
                   batch_tensor_input_ids=torch.tensor(batch_input_ids,dtype=torch.long)
                   batch_tensor_label_ids = torch.tensor(batch_label_ids,dtype=torch.long)
                   batch_cnt=0
                   batch_input_ids=[]
                   batch_label_ids=[]
                   yield (batch_tensor_input_ids,batch_tensor_label_ids)
           if len(batch_input_ids)>0:
               batch_tensor_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
               batch_tensor_label_ids = torch.tensor(batch_label_ids, dtype=torch.long)
               yield (batch_tensor_input_ids, batch_tensor_label_ids)

   def __iter__(self):
       worker_info=get_worker_info()
       if worker_info is None:
           for file in self.file_list:
               yield from  self.__parse_data(file)
       else:
           worker_number= worker_info.num_workers
           worker_id= worker_info.id
           for index,file in enumerate(self.file_list):
               if index%worker_number==worker_id:
                   yield from self.__parse_data(file)