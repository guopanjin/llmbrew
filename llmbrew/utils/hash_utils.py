import xxhash
import os
import random
import numpy as np
import torch

def xxhash_encoder(value,seed=10):
    value=str(value)
    return xxhash.xxh64(value,seed=seed).intdigest()

def set_seed(seed: int = 42, deterministic: bool = False):
    # Python built-in random
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch CPU
    torch.manual_seed(seed)
    # PyTorch GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Python hash seed (important for reproducibility)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic mode (trade speed for reproducibility)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    print(xxhash_encoder(123))