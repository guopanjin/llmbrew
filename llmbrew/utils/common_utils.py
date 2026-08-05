import os.path
from pathlib import Path
'''
vocab_size: int = 16000,
hidden_dim: int = 320,
num_heads: int = 5,
intermediate_dim: int = 864,
num_decoder_layers: int = 4,
'''
def get_num_parameters(vocab_size,
                       hidden_dim,
                       intermediate_dim,
                       num_decoder_layers,
                       ):
    embedding_parameter=vocab_size*hidden_dim
    decoder_layer_parameter=hidden_dim*hidden_dim*4 + hidden_dim*intermediate_dim*3
    return num_decoder_layers*decoder_layer_parameter + embedding_parameter

def get_project_path():
    abs_file_path=Path(__file__).resolve()
    project_path=abs_file_path.parents[2]
    return project_path

def resolve_path(path):
    abs_path=os.path.expanduser(path)
    if not os.path.isdir(abs_path):
        raise ValueError(f"This path:{path} is not existing")
    return abs_path
