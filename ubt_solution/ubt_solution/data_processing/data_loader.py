from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple
import logging
from .dataset import BehaviorSequenceDataset
from ..config import Config
from torch.utils.data._utils.collate import default_collate

logger = logging.getLogger(__name__)

def custom_collate(batch):
    """自定义 collate_fn，保留 cats_in_target 和 skus_in_target 的原始列表"""
    out = {}
    sample = batch[0]
    for k in sample:
        if k in ('cats_in_target', 'skus_in_target'):
            out[k] = [b[k] for b in batch]
        else:
            out[k] = default_collate([b[k] for b in batch])
    return out

def create_data_loaders(data_dir: Path, config: Config, mode: str = 'train', test_mode: bool = False):
    """根据mode创建不同的数据加载器"""
    if mode == 'train':
        train_dataset = BehaviorSequenceDataset(
            data_dir=data_dir,
            config=config,
            mode='train',
            test_mode=test_mode
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=custom_collate
        )
        return train_loader
    elif mode == 'valid':
        valid_dataset = BehaviorSequenceDataset(
            data_dir=data_dir,
            config=config,
            mode='infer',
            test_mode=True
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=custom_collate
        )
        return valid_loader
    elif mode == 'infer':
        infer_dataset = BehaviorSequenceDataset(
            data_dir=data_dir,
            config=config,
            mode='infer',
            test_mode=test_mode
        )
        infer_loader = DataLoader(
            infer_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=custom_collate
        )
        return infer_loader
    else:
        raise ValueError(f"Unknown mode: {mode}")

