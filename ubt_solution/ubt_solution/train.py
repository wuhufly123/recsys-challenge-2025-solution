import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import json
from datetime import datetime
from .model import UBTModel
from .data_processor import create_data_loaders
from .config import Config
from .trainer import UBTTrainer

logger = logging.getLogger(__name__)

def train(config: Config, data_dir: Path, output_dir: Path):
    """训练模型的主函数"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(data_dir, config)
    
    # 创建模型
    model = UBTModel(config).to(device)
    
    # 创建训练器
    trainer = UBTTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # 训练模型
    trainer.train()
    
    # 保存模型
    trainer.save_model(output_dir / "model.pt")
    
    return model

def main():
    # 加载配置
    config = Config()
    
    # 设置数据目录和输出目录
    data_dir = Path("data")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # 训练模型
    model = train(config, data_dir, output_dir)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main() 