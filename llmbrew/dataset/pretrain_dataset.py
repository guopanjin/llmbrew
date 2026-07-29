import os
import pandas as pd
import numpy as np
from torch.utils.data import IterableDataset,get_worker_info
from llmbrew.utils import Logger
'''
Build the streamingDataset based on the pytorch API
return (featureslabels)
'''
logger=Logger.get_logger()
class  PretrainDataset(IterableDataset):
   def __init__(self,data_path,
                context_length=512,
                batch_size=1000):
       self.data_path=os.path.expanduser(data_path)
       self.context_length=context_length
       self.batch_size=batch_size
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
   return (features,labels)==>([batch_size,seq_len],[batch_size,seq_len])
   '''
   def __parse_data(self,file):
       if ".bin" in file:
           np_data=np.memmap(file,dtype=np.uint16,mode="r")
           size=np_data.shape[0]
           read_size=0
           read_start_index=0
           logger.info(f"start to read data from {file},size={size}")
           while read_size<=size:
               read_end_index =read_start_index+ self.batch_size * (self.context_length+1)
               if read_end_index >size:#numpy slice does not include right side,so the end index can be size
                   logger.info(f"read_end_index >size-1 break,read_end_index:{read_end_index},size-1:{size-1}")
                   break;
               batch_data=np.array(np_data[read_start_index:read_end_index],dtype=np.int64,copy=True)
               batch_data=batch_data.reshape(self.batch_size,self.context_length+1)
               # update read_start_index,numpy slice does not include right side
               read_start_index = read_end_index
               read_size+=self.batch_size * (self.context_length+1)
               features=batch_data[:,:-1]
               labels=batch_data[:,1:]
               yield (features,labels)
   def __iter__(self):
       worker_info=get_worker_info()
       if worker_info is None:
           for file in self.file_list:
               yield from self.__parse_data(file)
       else:
           worker_number= worker_info.num_workers
           worker_id= worker_info.id
           for index,file in enumerate(self.file_list):
               if index%worker_number==worker_id:
                   yield from self.__parse_data(file)