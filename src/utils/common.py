import os
import random
import numpy as np
import torch
from typing import Union
from pathlib import Path

def seed_everything(seed: int) -> None:
    """
    固定所有随机种子，确保实验可复现。
    
    Args:
        seed (int): 随机种子数值
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 强制 CuDNN 使用确定性算法 (可能会降低一点速度，但结果可复现)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_dir(path: Union[str, Path]) -> None:
    """
    检查目录是否存在，不存在则递归创建。
    """
    if path is None:
        return
    Path(path).mkdir(parents=True, exist_ok=True)