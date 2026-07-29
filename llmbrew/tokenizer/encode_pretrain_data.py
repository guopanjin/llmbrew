from llmbrew.utils import xxhash_encoder
from llmbrew.config import data_config
import os
import json
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import numpy as np
from llmbrew.utils.log_utils import Logger

logger=Logger.get_logger()

train_portion = 99
validation_portion = 1

all_pretrain_data_paths = [data_config.llmbrew_pretrain_en_v1_path,
                      data_config.llmbrew_pretrain_zh_v1_path,
                      data_config.llmbrew_pretrain_code_v1_path
                      ]
train_bin_file = os.path.join(data_config.train_bin_path, "train.bin")
validation_bin_file = os.path.join(data_config.validation_bin_path, "validation.bin")
tokenizer_path = data_config.llmbrew_tokenizer_v1


def encode_data():
    num_trains = 0
    num_validation = 0
    global_number=0
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(os.path.expanduser(tokenizer_path), use_fast=True)
    logger.info(f"successfully load the tokenizer:{type(tokenizer)},path:{os.path.expanduser(tokenizer_path)}")
    eos_token_id=tokenizer.eos_token_id
    logger.info(f"eos_token_id:{eos_token_id}")
    #in case every single doc opens one file,so we put with open outside of for loop
    with open(os.path.expanduser(train_bin_file), "wb") as train_file,\
            open(os.path.expanduser(validation_bin_file), "wb") as validation_file:
        for data_path in all_pretrain_data_paths:
            data_path = os.path.expanduser(data_path)
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:  # streaming read line from local disk
                    raw_text = json.loads(line)["text"]
                    raw_text = raw_text.strip()
                    if len(raw_text) == 0:
                        continue
                    global_number+=1
                    if (global_number%1000)==0:
                        logger.info(f"global_number:{global_number}")
                    hash_id = xxhash_encoder(raw_text) % (train_portion + validation_portion)
                    tokenids = tokenizer.encode(raw_text, add_special_tokens=False)
                    tokenids.append(eos_token_id)
                    tokenid_array = np.array(tokenids, dtype=np.uint16)
                    if hash_id+1 > validation_portion:
                        num_trains += 1
                        tokenid_array.tofile(train_file)
                    else:
                        num_validation+=1
                        tokenid_array.tofile(validation_file)
    logger.info(f"finished encoding,num_trains:{num_trains},num_validation:{num_validation}")
'''
verify the bin file
'''
def verify_bin_data():
    data=np.memmap(os.path.expanduser(train_bin_file),
              dtype=np.uint16,
              mode="r"
              )
    logger.info(f"data_shape:{data.shape[0]}")
    token_size=data.shape[0]
    logger.info(f"token_size:{token_size}")
    tokenizer:PreTrainedTokenizerFast=AutoTokenizer.from_pretrained(os.path.expanduser(tokenizer_path),use_fast=True)
    sampleTokenids=data[2000:5000]
    logger.info(f"sampleTokenids shape:{sampleTokenids.shape[0]}")
    logger.info(f"sampleTokenIds:{sampleTokenids},type:{type(sampleTokenids)}")
    sampleTokenid_list=list(sampleTokenids)
    decoded = tokenizer.decode(sampleTokenid_list,
                                  skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False
                                  )
    print(repr(decoded))


if __name__ == '__main__':
    #encode_data()
    verify_bin_data()
    pass