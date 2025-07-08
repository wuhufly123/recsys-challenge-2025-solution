from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch
import logging
import os
import argparse

logger = logging.getLogger(__name__)

@dataclass
class Config:
    # 模型架构参数
    hidden_size: int = 256  # 增加隐藏层大小以提升模型表达能力
    num_heads: int = 4  # 注意力头数量
    num_layers: int = 3  # HSTU层数
    dropout: int = 0.2  # dropout比例
    
    # HSTU特定参数
    attention_dim: int = 128  # 注意力维度
    linear_dim: int = 256  # 线性层维度
    linear_activation: str = "silu"  # 线性层激活函数
    normalization: str = "rel_bias"  # 归一化方法
    linear_config: str = "uvqk"  # 线性层配置
    concat_ua: bool = False  # 是否连接u和a
    time_buckets: int = 128  # 时间桶数量
    
    # 特征维度
    num_categories: int = 10000
    num_candidates_categories: int = 100
    num_products: int = 100  # 添加产品数量参数
    num_behaviors: int = 6  # 购买、加购、移除、页面访问、搜索
    name_vector_dim: int = 16
    query_vector_dim: int = 16
    
    # 新增URL和Item ID特征配置
    url_hash_size: int = 100000  # URL哈希空间大小
    item_embedding_dim: int = 128  # Item ID嵌入维度
    url_embedding_dim: int = 128   # URL嵌入维度
    max_item_id: int = 2000000    # 最大Item ID数量
    
    # 添加：SKU哈希空间大小
    sku_hash_size: int = 10000    # 将原始SKU映射到此哈希空间
    
    # 行为类型配置
    max_seq_length: int = 300  # 序列长度
    
    # 训练配置
    batch_size: int = 8192  # 批次大小
    learning_rate: float = 5e-5  # 学习率
    weight_decay: float = 1e-3  # 权重衰减
    num_epochs: int = 150  # 训练轮数
    warmup_steps: int = 2000  # 预热步数
    patience: int = 10  # 早停耐心
    
    # 设备配置
    accelerator: str = "cuda"
    devices: List[int] = field(default_factory=lambda: [0])
    num_workers: int = 10
    # 负采样数量，用于负采样softmax-ce损失
    num_negative_samples: int = 400
    output_dir: str = "./outputs"
    device: str = "cuda:0"
    use_cpu: bool = False  # 添加CPU选项
    
    # 多任务学习配置
    task_weights: Dict[str, float] = field(default_factory=lambda: {
        'churn': 1.0,
        'category_propensity': 0.5,
        'product_propensity': 0.5,
        'price': 1.0,
        'name': 0.5  # 下一个购买商品名称预测权重
    })
    
    # 稳定性参数
    gradient_norm_clip: float = 1.0  # 梯度范数裁剪值
    embedding_dropout: float = 0.1  # embedding层的dropout
    attention_dropout: float = 0.1  # 注意力的dropout
    relu_dropout: float = 0.1  # ReLU激活后的dropout
    residual_dropout: float = 0.1  # 残差连接的dropout
    
    # 损失函数参数
    loss_scale: float = 0.1  # 损失缩放因子
    pos_weight: float = 5.0  # 正样本权重
    # 新增：Novelty 损失权重，用于平衡 novelty loss
    novelty_weight: float = 0.5  # Novelty loss 在总损失中的权重
    use_dynamic_task_weights: bool = False  # 是否使用动态任务权重
    
    # 新增：Focal Loss 和倾向性任务样本加权参数
    focal_loss_gamma: float = 2.0  # Focal Loss 的 gamma 参数
    focal_loss_alpha: float = 0.75 # Focal Loss 的 alpha 参数 (给正样本的权重，如果正样本稀疏，alpha > 0.5)
    propensity_positive_sample_weight_boost: float = 5.0 # 对有真实标签的倾向性样本的额外权重提升值
    propensity_negative_sample_weight: float = 0.2 # 对无真实标签的样本赋予较小权重
    
    padding_idx: Dict[str, int] = field(default_factory=lambda: {
        'category': 0, 
        'price': 0,
        # Add other features here if they also need specific padding indices for embedding layers
    })
    
    # 价格最大值，用于将 sigmoid 输出映射到实际区间 [0, max_price]
    max_price: int = 100
    
    def __post_init__(self):
        if self.task_weights is None:
            self.task_weights = {
                'churn': 1.0,
                'category_propensity': 0.8,
                'product_propensity': 0.8
            }
            
        # 检查 hidden_size 是否能被 num_heads 整除
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})")
            
        # 检查 CUDA 可用性
        if self.use_cpu or not torch.cuda.is_available():
            logger.warning("Using CPU for training")
            self.accelerator = "cpu"
            self.device = "cpu"
        else:
            try:
                # 尝试创建一个小的CUDA张量来测试CUDA是否正常工作
                test_tensor = torch.zeros(1, device="cuda")
                del test_tensor
                
                # 确保设备字符串格式正确
                if not self.device.startswith("cuda:"):
                    self.device = f"cuda:{self.devices[0]}"
                
                # 检查 CUDA 版本
                cuda_version = torch.version.cuda
                logger.info(f"Using CUDA version: {cuda_version}")
                
                if cuda_version < "11.0":
                    logger.warning(f"CUDA version {cuda_version} may be too old. Recommended: 11.0 or higher")
            except Exception as e:
                logger.warning(f"CUDA initialization failed: {e}. Falling back to CPU")
                self.accelerator = "cpu"
                self.device = "cpu"

# 创建默认配置实例
config = Config()

