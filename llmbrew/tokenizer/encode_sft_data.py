from llmbrew.tokenizer import TokenizerUtil
from llmbrew.config import data_config
import pandas as pd
from llmbrew.constants import Constant
import json
from llmbrew.utils import set_seed
import random
import os
from llmbrew.utils import Logger
logger=Logger.get_logger()
train_path=data_config.llmbrew_sft_train_path
validation_path=data_config.llmbrew_sft_validation_path

tokenizer=TokenizerUtil()

set_seed(seed=42)
'''
columns for en_file:'instruction', 'input', 'output', 'text'
    PAD="<pad>",
    BOS="<bos>",
    EOS="<eos>",
    SYSTEM="<system>",
    USER="<user>",
    ASSISTANT="<assistant>",
    UNK="<unk>"
Drop data that the len is great then 512.If we clip the line to 512.The model will learn the clipping information.
'''
def build_jsonl(data:list[dict],source)->list[dict]:
    num_drop_data=0
    jsonl=[]
    for row in data:
        system = f"{Constant.SpecialToken.SYSTEM}you are a helpful assistant."
        instruction = row["instruction"]
        input = row["input"]
        output = row["output"].strip()
        prompt = system + f"{Constant.SpecialToken.USER}{instruction}" + f"{input}{Constant.SpecialToken.ASSISTANT}"
        response = output + Constant.SpecialToken.EOS
        line = prompt + response
        tokenids = tokenizer.encode(line)
        prompt_len = len(tokenizer.encode(prompt))
        output_json = {
            "input_ids": tokenids,
            "prompt_len": prompt_len,
            "source": source,
            "prompt": prompt,
            "response": response
        }
        if len(tokenids) > 512:
            num_drop_data += 1
            continue
        jsonl.append(output_json)
    num_valid = len(jsonl)
    print(f"num_drop_data:{num_drop_data}, num_valid:{num_valid}")
    return jsonl
'''
num_drop_data:110, num_valid:51892
'''
def read_en_data():
    en_df = pd.read_parquet(data_config.llmbrew_sft_raw_data_en_file)
    data=[]
    for _, row in en_df.iterrows():
        data.append({
            "instruction":row["instruction"],
            "input": row["input"],
            "output":row["output"].strip()
        })
    jsonl=build_jsonl(data,source="en")
    return jsonl
'''
en   51,892   drop 110  (0.21%)
zh   48,684   drop 134  (0.27%)
     100,576
'''
def read_zh_data():
    with open(data_config.llmbrew_sft_raw_data_zh_file,"r",encoding="utf-8") as f:
        zh_list_json=json.load(f)
    jsonl = build_jsonl(zh_list_json,source="zh")
    return jsonl

def shuffle_data():
    en_jsonl = read_en_data()
    zh_jsonl = read_zh_data()
    all_jsonl=en_jsonl+zh_jsonl
    random.shuffle(all_jsonl)
    train_jsonl=all_jsonl[1000:]
    validation_jsonl=all_jsonl[0:1000]
    logger.info(f"train_jsonl_cnt:{len(train_jsonl)},validation_jsonl_cnt:{len(validation_jsonl)}")
    os.makedirs(train_path,exist_ok=True)
    os.makedirs(validation_path, exist_ok=True)
    with open(train_path+"train.jsonl","w",encoding="utf-8") as f:
        for json_data in train_jsonl:
            json_str=json.dumps(json_data,ensure_ascii=False) #ensure the Chinese character is readable
            f.write(json_str+"\n")
    with open(validation_path+"validation.jsonl","w",encoding="utf-8") as f:
        for json_data in validation_jsonl:
            json_str=json.dumps(json_data,ensure_ascii=False)
            f.write(json_str+"\n")

if __name__ == '__main__':
    #jsonl=read_en_data()
    #print(jsonl[0:3])
    #jsonl=read_zh_data()
    #print(jsonl[0:3])
    shuffle_data()
    pass

