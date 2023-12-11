import logging
import os
from datetime import datetime

def log_training_info(args_dict, test_acc, test_macro, test_loss, log_dir):
    # 创建一个日志文件的名称
    log_file = os.path.join(log_dir, "train_log.txt")

    # 创建一个日志记录器
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 创建一个日志处理器，将日志写入到一个文件中
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)

    # 将日志处理器添加到日志记录器中
    logger.addHandler(handler)

    # 记录训练结果和超参数
    logger.info(datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3])  # 记录当前时间
    logger.info(f"="*100)
    logger.info(f"Training Parameters: ")
    for key, value in args_dict.items():
        logger.info(f"    {key}: {value}")  # 记录args_dict中的参数
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Test Macro F1: {test_macro:.4f}")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"="*100 + "\n") 