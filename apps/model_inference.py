from llmbrew.constants.constant import Constant
from llmbrew.utils.log_utils import Logger
from llmbrew.config import data_config,model_artifacts_config
from llmbrew.model import LLMBrewModel,LLMBrewConfig
from llmbrew.utils import set_seed
from llmbrew.tokenizer import TokenizerUtil
import torch
import os
logger=Logger.get_logger()
set_seed()
DEMO_PROMPTS = [
     "你是谁?",
    "How can I stay healthy?",
    "Write a short introduction about yourself.",
]
model_file_path=model_artifacts_config.model_outputs_sft_final_model_path+"sft_model.pt"
model_states=torch.load(os.path.expanduser(model_file_path),weights_only=False)
model_config=model_states["config"].__dict__
num_tokens=model_states["num_tokens"]
model_state_dict=model_states["model_state_dict"]
logger.info(f"num_tokens:{num_tokens}")
logger.info(model_config)
logger.info(f"load model from {model_file_path}")

llmbrewconfig= LLMBrewConfig(**model_config)
model=LLMBrewModel(llmbrewconfig=llmbrewconfig)
model.load_state_dict(state_dict=model_state_dict)
model=model.to("cpu")


tokenizer_path=data_config.llmbrew_tokenizer_v1
tokenizer=TokenizerUtil(tokenizer_path=tokenizer_path)
logger.info(f"load tokenizer from {tokenizer_path}")


def generate_stream():
    for prompt in DEMO_PROMPTS:
        system = f"{Constant.SpecialToken.SYSTEM}you are a helpful assistant."
        final_prompt = system + f"{Constant.SpecialToken.USER}{prompt}{Constant.SpecialToken.ASSISTANT}"
        model.eval()
        tokenids = tokenizer.encoder(final_prompt)
        if not tokenids:  # in case null value
            tokenids.append(2)
        tokenids = torch.tensor(tokenids, dtype=torch.long)
        '''
        input:[batch_size,inputids] 
        return:[batch_size,max_new_tokens]
        '''
        tokenids = torch.unsqueeze(tokenids, dim=0)

        output_generator = model.generate_stream(input_ids=tokenids,
                                    do_sample=True,
                                    temperature=0.8,
                                    top_k=40,
                                    max_new_tokens=100,
                                    eos_token_id=2
                                    )
        acc_tokenid=[]
        pre=""
        print(f"prompt:{final_prompt}")
        for output_id in output_generator:
            acc_tokenid.append(output_id)
            output_token=tokenizer.deconder(acc_tokenid)
            print(output_token[len(pre):],end="",flush=True)
            pre=output_token
        print()
def generate_stream_for_chatbox(prompt,
                                do_sample=True,
                                temperature=0.8,
                                top_k=40,
                                max_new_tokens=100,
                                eos_token_id=2
                                ):
    logger.info(f"generator parameter do_sample:{do_sample},temperature:{temperature},top_k:{top_k},max_new_tokens:{max_new_tokens},eos_token_id:{eos_token_id}")
    system = f"{Constant.SpecialToken.SYSTEM}you are a helpful assistant."
    final_prompt = system + f"{Constant.SpecialToken.USER}{prompt}{Constant.SpecialToken.ASSISTANT}"
    model.eval()
    tokenids = tokenizer.encoder(final_prompt)
    if not tokenids:  # in case null value
        tokenids.append(2)
    tokenids = torch.tensor(tokenids, dtype=torch.long)
    '''
    input:[batch_size,inputids] 
    return:[batch_size,max_new_tokens]
    '''
    tokenids = torch.unsqueeze(tokenids, dim=0)

    output_generator = model.generate_stream(input_ids=tokenids,
                                do_sample=do_sample,
                                temperature=temperature,
                                top_k=top_k,
                                max_new_tokens=max_new_tokens,
                                eos_token_id=eos_token_id
                                )
    acc_tokenid=[]
    pre = ""
    print(f"prompt:{final_prompt}")
    for output_id in output_generator:
        acc_tokenid.append(output_id)
        output_token = tokenizer.deconder(acc_tokenid)
        new_token=output_token[len(pre):]
        pre = output_token
        yield new_token


if __name__ == '__main__':
    generator=generate_stream_for_chatbox("How can I stay healthy?")
    for content in generator:
        print(content)
