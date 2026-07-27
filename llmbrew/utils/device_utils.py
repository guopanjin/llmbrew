import sys

import torch
import subprocess
import atexit
import shutil
import os
from llmbrew.utils.log_utils import  Logger
logger=Logger.get_logger()


def get_device(local_rank=None):
    if local_rank is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def prevent_sleep()->None:
    #darwin is the name of mac os
    if sys.platform !="darwin":
        return None
    if shutil.which("caffeinate") is None:
        return None
    #start a subprocess and it will not block the main process,return subprocess
    #-i = idle, -s = system sleep -w = wait
    process=subprocess.Popen(["caffeinate","-is","-w",str(os.getpid())],
                     stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL
                     )
    #main process exit,close the sub process
    #register needs a function. pass the method itself,not its result,atexit call it later
    atexit.register(process.terminate)
    logger.info(f"caffeinate started {process.pid},sleeping prevention active")
