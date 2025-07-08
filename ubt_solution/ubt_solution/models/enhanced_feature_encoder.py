import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init # Added init for _initialize_weights
from typing import Dict, List, Optional, Tuple # Though not directly used by this class, good practice if it were
from ..config import Config
import logging

logger = logging.getLogger(__name__)

class EnhancedFeatureEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # 事件类型编码
        self.event_embedding = nn.Sequential(
            nn.Embedding(5, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout)
        )
        
        # 商品特征编码
        self.category_embedding = nn.Sequential(
            nn.Embedding(config.num_categories + 1, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout)
        )
        
        self.price_embedding = nn.Sequential(
            nn.Embedding(101, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout)
        )
        
        # 名称和查询编码：EmbeddingBag 平均 16 桶 id
        self.name_embedding = nn.EmbeddingBag(256, config.hidden_size, mode='mean', padding_idx=0)
        self.query_embedding = nn.EmbeddingBag(256, config.hidden_size, mode='mean', padding_idx=0)
        self.name_ln = nn.LayerNorm(config.hidden_size)
        self.query_ln = nn.LayerNorm(config.hidden_size)
        self.name_dropout = nn.Dropout(config.dropout)
        self.query_dropout = nn.Dropout(config.dropout)
        
        # 新增：Item ID编码 - 使用哈希映射
        self.item_embedding = nn.Sequential(
            nn.Embedding(config.sku_hash_size, config.item_embedding_dim, padding_idx=0),
            nn.LayerNorm(config.item_embedding_dim),
            nn.Dropout(config.dropout)
        )
        
        # 新增：URL编码 - 使用哈希技巧映射到较小的空间
        self.url_embedding = nn.Sequential(
            nn.Embedding(config.url_hash_size, config.url_embedding_dim, padding_idx=0),
            nn.LayerNorm(config.url_embedding_dim),
            nn.Dropout(config.dropout)
        )
        
        # URL和Item ID特征映射到隐藏层大小
        self.item_projection = nn.Sequential(
            nn.Linear(config.item_embedding_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        self.url_projection = nn.Sequential(
            nn.Linear(config.url_embedding_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # 特征重要性评估 - 使用更稳定的结构
        self.importance_net = nn.Sequential(
            nn.Linear(config.hidden_size, 64),
            nn.LayerNorm(64),  # 添加归一化层
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 特征融合 - 更新融合维度以包含新增特征
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 7, config.hidden_size * 2),  # 移除时间特征, 7 个特征
            nn.LayerNorm(config.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Tanh()  # 使用tanh限制输出范围在[-1,1]
        )
        
        # 新增：name 嵌入的 MLP 映射
        self.name_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用更保守的初始化方法
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)  # 减小初始权重方差
                
    # URL哈希函数 - 将大范围的URL ID映射到较小的哈希空间
    def hash_url(self, url_ids):
        # 使用简单的模运算进行哈希，保证值在哈希空间内
        return url_ids % self.config.url_hash_size

    def forward(self, 
                event_types: torch.Tensor,
                categories: torch.Tensor,
                prices: torch.Tensor,
                names: torch.Tensor,
                queries: torch.Tensor,
                timestamps: torch.Tensor,  # timestamps 保留 规范时间，无线性映射
                item_ids: torch.Tensor = None,  # 新增参数
                urls: torch.Tensor = None) -> torch.Tensor:  # 新增参数
        batch_size, seq_len, _ = names.shape
        
        # 确保索引在有效范围内并且没有NaN值
        event_types = torch.clamp(event_types, 0, 4)
        categories = torch.clamp(categories, 0, self.config.num_categories)
        prices = torch.clamp(prices, 0, 100)
        
        # 检查并清理NaN值
        if torch.isnan(names).any():
            names = torch.where(torch.isnan(names), torch.zeros_like(names), names)
        if torch.isnan(queries).any():
            queries = torch.where(torch.isnan(queries), torch.zeros_like(queries), queries)
        if torch.isnan(timestamps).any():
            timestamps = torch.where(torch.isnan(timestamps), torch.zeros_like(timestamps), timestamps)
        
        # 处理item_ids，确保在有效范围内
        if item_ids is not None:
            # 限制在hash空间内
            item_ids = torch.clamp(item_ids, 0, self.config.sku_hash_size - 1)
            
        # 处理urls，进行哈希并确保有效
        if urls is not None:
            # 确保没有负值
            urls = torch.where(urls < 0, torch.zeros_like(urls), urls)
            # 哈希URL ID
            hashed_urls = self.hash_url(urls)
        else:
            # 如果未提供URLs，使用全零张量
            hashed_urls = torch.zeros_like(event_types)
            
        # 规范化时间戳，无独立线性映射，由相对偏置处理
        
        # 编码事件类型
        event_emb = self.event_embedding(event_types)
        
        # 编码商品特征
        cat_emb = self.category_embedding(categories)
        price_emb = self.price_embedding(prices)
        
        # 编码名称和查询
        names_ids = names.long().view(-1, 16)
        name_emb_2d = self.name_embedding(names_ids)
        name_emb = name_emb_2d.view(batch_size, seq_len, self.config.hidden_size)
        name_emb = self.name_ln(name_emb)
        name_emb = self.name_dropout(name_emb)
        
        # 新增：通过 MLP 提升 name 特征表达
        name_emb = self.name_mlp(name_emb)
        
        queries_ids = queries.long().view(-1, 16)
        query_emb_2d = self.query_embedding(queries_ids)
        query_emb = query_emb_2d.view(batch_size, seq_len, self.config.hidden_size)
        query_emb = self.query_ln(query_emb)
        query_emb = self.query_dropout(query_emb)
        
        # 编码Item ID
        if item_ids is not None:
            item_emb = self.item_embedding(item_ids)
            item_emb = self.item_projection(item_emb)
        else:
            # 如果未提供Item ID，使用全零嵌入
            item_emb = torch.zeros(batch_size, seq_len, self.config.hidden_size, device=event_types.device)
            
        # 编码URL
        url_emb = self.url_embedding(hashed_urls)
        url_emb = self.url_projection(url_emb)
        
        # 计算特征重要性 - 添加数值稳定性 - 现在包含Item ID和URL特征
        features = [event_emb, cat_emb, price_emb, name_emb, query_emb, item_emb, url_emb]
        importance_scores = []
        for feat in features:
            # 添加特征归一化
            norm_feat = F.normalize(feat, p=2, dim=-1)
            score = self.importance_net(norm_feat)
            importance_scores.append(score)
        
        # 加权特征融合 - 使用softmax确保权重和为1
        all_scores = torch.cat(importance_scores, dim=-1)
        normalized_scores = F.softmax(all_scores, dim=-1).unsqueeze(-1)
        
        weighted_features = []
        for i, feat in enumerate(features):
            weighted_features.append(feat * normalized_scores[:,:,i])
        
        # 特征融合
        features = torch.cat(weighted_features, dim=-1)
        fused_features = self.feature_fusion(features)
        
        # 检查并处理NaN值
        if torch.isnan(fused_features).any():
            logger.warning("NaN values detected in feature fusion output. Replacing with zeros.")
            fused_features = torch.where(
                torch.isnan(fused_features), 
                torch.zeros_like(fused_features), 
                fused_features
            )
        
        # 添加额外的梯度裁剪
        fused_features = torch.clamp(fused_features, min=-5.0, max=5.0)
        
        return fused_features 