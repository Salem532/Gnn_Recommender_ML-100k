import logging
import sys
import os
from src.utils.common import check_dir

def setup_logger(name: str, save_dir: str = "", filename: str = "train.log") -> logging.Logger:
    """
    配置并返回一个全局 Logger 对象。
    
    Args:
        name (str): Logger 的名称
        save_dir (str): 日志文件保存目录
        filename (str): 日志文件名
        
    Returns:
        logging.Logger: 配置好的 Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 避免重复添加 Handler
    if logger.handlers:
        return logger

    # 格式化: 时间 - 级别 - 消息
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # 1. 控制台输出 Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 2. 文件输出 Handler (如果有路径)
    if save_dir:
        check_dir(save_dir)
        fh = logging.FileHandler(os.path.join(save_dir, filename), mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger