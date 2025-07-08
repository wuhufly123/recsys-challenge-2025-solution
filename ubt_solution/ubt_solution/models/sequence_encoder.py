import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from typing import Optional, Tuple
from ..config import Config
from .positional_encoding import PositionalEncoding
import logging

logger = logging.getLogger(__name__)

class SequenceEncoder(nn.Module):
    def __init__(self, config: Config): # Added type hint for config
        super().__init__()
        self.config = config
        
        # 位置编码
        self.position_encoding = PositionalEncoding(config.hidden_size, config.max_seq_len if hasattr(config, 'max_seq_len') else 5000) # Pass max_len to PositionalEncoding
        
        # Transformer编码器 - 使用更稳定的配置
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 2,  # 减小前馈网络维度
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,  # 先归一化再做注意力，提高稳定性
            activation='gelu'
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        # 序列池化 - 使用多种池化方式然后融合
        self.attention_pooling = nn.Sequential(
            nn.Linear(config.hidden_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Tanh()  # 使用tanh限制输出范围
        )
        
        # 用户表示
        self.user_representation = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Tanh()  # 限制输出范围
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # 保存原始输入用于残差连接
        residual = x
        
        # 位置编码
        x = self.position_encoding(x)
        
        # 转换 mask 格式
        if mask is not None:
            if mask.dim() < 2:
                mask = mask.unsqueeze(0)
            
            # 创建正确的注意力掩码
            src_key_padding_mask = ~mask
        else:
            src_key_padding_mask = None
        
        # 添加输入归一化
        x = F.normalize(x, p=2, dim=-1) * math.sqrt(self.config.hidden_size)
        
        # Transformer编码
        temporal_features = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # 残差连接 - 使用缩放因子
        temporal_features = temporal_features + residual * 0.1
        
        # 序列池化 - 平均池化 (带掩码)
        if mask is not None:
            mean_pooled = torch.sum(temporal_features * mask.unsqueeze(-1), dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
        else:
            mean_pooled = torch.mean(temporal_features, dim=1)
        
        # 注意力池化
        attention_weights = self.attention_pooling(temporal_features)
        if mask is not None:
            # 掩蔽填充位置
            attention_weights = attention_weights * mask.unsqueeze(-1)
            # 重新归一化权重
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
        attention_pooled = torch.sum(attention_weights * temporal_features, dim=1)
        
        # 融合不同的池化结果
        pooled = self.fusion_layer(torch.cat([mean_pooled, attention_pooled], dim=-1))
        
        # 用户表示
        user_embedding = self.user_representation(pooled)
        
        # 检查并处理NaN值
        if torch.isnan(user_embedding).any():
            logger.warning("NaN values detected in user embedding. Replacing with zeros.")
            user_embedding = torch.where(
                torch.isnan(user_embedding), 
                torch.zeros_like(user_embedding), 
                user_embedding
            )
        
        # 裁剪值范围
        user_embedding = torch.clamp(user_embedding, min=-5.0, max=5.0)
        temporal_features = torch.clamp(temporal_features, min=-5.0, max=5.0)
        
        return user_embedding, temporal_features 