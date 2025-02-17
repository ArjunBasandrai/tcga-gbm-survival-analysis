import numpy as np
import torch
import random
import os

def seed_env():
    np.random.seed(int(os.environ['PROJECT_SEED']))
    random.seed(int(os.environ['PROJECT_SEED']))
    torch.manual_seed(int(os.environ['PROJECT_SEED']))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(os.environ['PROJECT_SEED']))
        torch.cuda.manual_seed_all(int(os.environ['PROJECT_SEED']))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_env()