import random
import torch
import numpy as np

def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def parse_range(range_str:str, separator:str=","):
    
    start, end = None, None

    if range_str and range_str.count(separator) == 1:
        start, end = map(str.strip, range_str.split(separator))

        start = int(start) if start else None
        end = int(end) if end else None
    return start, end
