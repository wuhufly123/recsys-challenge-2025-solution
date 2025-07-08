import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from ..config import Config
from .enhanced_feature_encoder import EnhancedFeatureEncoder
from .task_specific_encoder import TaskSpecificEncoder
from .hstu_modules import (
    RelativeBucketedTimeAndPositionBasedBias,
    SequentialTransductionUnitJagged,
    HSTUJagged,
    HSTUCacheState
)
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Focal Loss 实现
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: logits (B, C) where C is number of classes
        targets: binary ground truth (B, C)
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Probs of correct class
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else: # 'none'
            return F_loss

class UniversalBehavioralTransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.feature_encoder = EnhancedFeatureEncoder(config)
        
        # 加载 active_clients，用于限制 churn 损失计算
        active_clients_np = np.load('/data_sdc/mhwang/recsys_code/data/target/active_clients.npy')
        self.register_buffer('active_client_ids', torch.from_numpy(active_clients_np).long())
        
        # Define Relative Attention Bias module once
        self.relative_attention_bias = RelativeBucketedTimeAndPositionBasedBias(
            max_seq_len=config.max_seq_length,
            num_buckets=config.time_buckets,
            bucketization_fn=lambda x: (
                torch.log(torch.abs(x).clamp(min=1)) / 0.69314718056 # Using ln(2)
            ).long(),
        )
        
        self.register_buffer(
            "_attn_mask",
            torch.triu(
                torch.ones(
                    (config.max_seq_length, config.max_seq_length),
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )
        
        self.hstu = HSTUJagged(
            modules=[
                SequentialTransductionUnitJagged(
                    embedding_dim=config.hidden_size,
                    linear_hidden_dim=config.linear_dim,  # dv per head
                    attention_dim=config.attention_dim,   # dqk per head
                    num_heads=config.num_heads,
                    relative_attention_bias_module=self.relative_attention_bias, # Pass the shared module
                    dropout_ratio=config.dropout, # General dropout
                    attn_dropout_ratio=config.attention_dropout, # Specific attention dropout
                    linear_activation=config.linear_activation,
                    # normalization and linear_config are less critical now
                )
                for _ in range(config.num_layers)
            ],
            autocast_dtype=None, # Consider config.mixed_precision_dtype if you add it
        )
        
        self.task_encoder = TaskSpecificEncoder(config)
        
        self.register_buffer('task_weights', torch.tensor([
            config.task_weights['churn'],
            config.task_weights['category_propensity'],
            config.task_weights['product_propensity']
        ]))
        
        self.loss_scale = config.loss_scale if hasattr(config, 'loss_scale') else 0.1
        self.pos_weight = config.pos_weight if hasattr(config, 'pos_weight') else 5.0
        self.use_dynamic_task_weights = config.use_dynamic_task_weights if hasattr(config, 'use_dynamic_task_weights') else False
        # 回归任务 price 权重
        self.price_weight = config.task_weights.get('price', 1.0)
        # 下一个购买商品名称预测损失权重，由 task_weights['name'] 控制
        self.name_weight = config.task_weights.get('name', 1.0)

        # 初始化 Focal Loss
        self.focal_loss = FocalLoss(
            alpha=config.focal_loss_alpha,
            gamma=config.focal_loss_gamma,
            reduction='none'  # We need per-sample losses for custom weighting
        )
        self.propensity_positive_sample_weight_boost = config.propensity_positive_sample_weight_boost

        # 负采样数量，默认为 10
        self.num_negative_samples = getattr(config, 'num_negative_samples', 10)
        # NCE 嵌入：使用全域 embedding
        # 类别 NCE 嵌入表：共享 feature_encoder.category_embedding 的 embedding 层
        self.category_embeddings = self.feature_encoder.category_embedding[0]
        # 商品 NCE 嵌入：共享 feature_encoder.item_embedding 和 item_projection
        self.sku_embeddings = nn.Sequential(
            self.feature_encoder.item_embedding[0],
            self.feature_encoder.item_projection
        )
        # 新增：名称预测 Head，将 user embedding 映射到 name_vector_dim
        self.name_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.name_vector_dim),
            nn.LayerNorm(config.name_vector_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        # 用户全局特征投影
        self.user_feat_proj = nn.Sequential(
            nn.Linear(11, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        feature_embeddings = self.feature_encoder(
            event_types=batch['event_types'],
            categories=batch['categories'],
            prices=batch['prices'],
            names=batch['names'],
            queries=batch['queries'],
            timestamps=batch['timestamps'],
            item_ids=batch['item_ids'],
            urls=batch['urls']
        )
        
        # DLRMv3式融合：将静态 user_feats 投影后加到每个时间步的 embedding
        if 'user_feats' in batch:
            ufeat = self.user_feat_proj(batch['user_feats'])  # [B, H]
            feature_embeddings = feature_embeddings + ufeat.unsqueeze(1)  # 广播到 [B, T, H]
        
        batch_size, seq_len, hidden_size = feature_embeddings.shape
        
        seq_lengths = batch['mask'].sum(dim=1).to(torch.int32)
        
        x_offsets = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        x_offsets[1:] = torch.cumsum(seq_lengths, dim=0)
        
        if batch['timestamps'].dtype != torch.int64:
            all_timestamps = (batch['timestamps'] * 86400).to(torch.int64)
        else:
            all_timestamps = batch['timestamps']
        
        causal_mask = 1.0 - self._attn_mask[:seq_len, :seq_len].float()
        
        
        invalid_attn_mask = causal_mask
        
        user_embeddings, cache_states = self.hstu(
            x=feature_embeddings,
            x_offsets=x_offsets,
            all_timestamps=all_timestamps,
            invalid_attn_mask=invalid_attn_mask,
        )
        # L2 normalize per-step sequence embeddings
        user_embeddings = F.normalize(user_embeddings, p=2, dim=-1)
        temporal_features = user_embeddings
        
        # 矢量化 last+avg 池化，去除 Python 循环
        lengths = seq_lengths.long()  # 转为 int64，保证 gather index 合法  [B]
        # 取最后一步输出
        idx = (lengths - 1).unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, hidden_size)
        last = torch.gather(user_embeddings, dim=1, index=idx).squeeze(1)  # [B, H]
        # 平均池化
        mask_f = batch['mask'].float().unsqueeze(-1)  # [B, T, 1]
        sum_emb = (user_embeddings * mask_f).sum(dim=1)  # [B, H]
        avg = sum_emb / lengths.unsqueeze(1)  # [B, H]
        # 融合
        user_embeddings = 0.5 * (last + avg)
        # L2 normalize pooled user embedding
        user_embeddings = F.normalize(user_embeddings, p=2, dim=-1)
        
        task_outputs = self.task_encoder(user_embeddings)
        
        # price 回归映射已移除，用 price_propensity 多标签分类代替
        
        losses = {}
        
        # churn 任务损失：仅对 active clients 计算
        client_ids = batch['client_id']
        active_mask = torch.isin(client_ids, self.active_client_ids)
        churn_logits = task_outputs['churn'].squeeze()
        churn_targets = batch['churn'].float()
        pos_w = torch.tensor(self.pos_weight, device=device) if churn_targets.sum() > 0 else None
        per_sample_churn_loss = F.binary_cross_entropy_with_logits(
            churn_logits, churn_targets, pos_weight=pos_w, reduction='none'
        )
        if active_mask.any():
            churn_loss = per_sample_churn_loss[active_mask].mean()
        else:
            churn_loss = torch.tensor(0.0, device=device)
        losses['churn_loss'] = churn_loss
        
        self.temperature = 0.07
        # 多正样本 NCE 损失：类别
        category_losses = []
        for i, user_emb in enumerate(user_embeddings):
            pos_ids = batch['cats_in_target'][i]
            if len(pos_ids) == 0:
                continue
            pos_tensor = torch.tensor(pos_ids, dtype=torch.long, device=device)
            neg_ids = batch['neg_cat_ids'][i]
            sampled_ids = torch.cat([pos_tensor, neg_ids], dim=0)
            emb = self.category_embeddings(sampled_ids)
            emb = F.normalize(emb, p=2, dim=-1)
            logits = (emb * user_emb.unsqueeze(0)).sum(-1) / self.temperature
            log_probs = torch.log_softmax(logits, dim=0)
            category_losses.append(-log_probs[:pos_tensor.size(0)].mean())
        if category_losses:
            category_loss = torch.stack(category_losses).mean()
        else:
            category_loss = torch.tensor(0.0, device=device)
        losses['category_loss'] = category_loss
        # 多正样本 NCE 损失：商品
        product_losses = []
        for i, user_emb in enumerate(user_embeddings):
            pos_ids = batch['skus_in_target'][i]
            if len(pos_ids) == 0:
                continue
            pos_tensor = torch.tensor(pos_ids, dtype=torch.long, device=device)
            neg_ids = batch['neg_sku_ids'][i]
            sampled_ids = torch.cat([pos_tensor, neg_ids], dim=0)
            emb = self.sku_embeddings(sampled_ids)
            emb = F.normalize(emb, p=2, dim=-1)
            logits = (emb * user_emb.unsqueeze(0)).sum(-1) / self.temperature
            log_probs = torch.log_softmax(logits, dim=0)
            product_losses.append(-log_probs[:pos_tensor.size(0)].mean())
        if product_losses:
            product_loss = torch.stack(product_losses).mean()
        else:
            product_loss = torch.tensor(0.0, device=device)
        losses['product_loss'] = product_loss
        # 价格回归损失（MSE），仅对有标签样本计算
        price_pred = task_outputs['price']  # 预测值，形状 [B]
        price_target = batch['price_target']
        has_mask = batch['has_price_target']
        mask = has_mask > 0.5
        if mask.any():
            price_loss = F.mse_loss(price_pred[mask], price_target[mask])
        else:
            price_loss = torch.tensor(0.0, device=device)
        losses['price_loss'] = price_loss
        weighted_price = self.price_weight * price_loss
        # 加入各任务的加权损失
        cw, catw, pw = self.task_weights
        weighted_churn = cw * churn_loss
        weighted_category = catw * category_loss
        weighted_product = pw * product_loss
        losses['weighted_churn_loss'] = weighted_churn
        losses['weighted_category_loss'] = weighted_category
        losses['weighted_product_loss'] = weighted_product
        losses['weighted_price_loss'] = weighted_price
        # 添加：Novelty 损失计算
        # 类别 Novelty loss：propensity_category_ids/popularity_category 对所有用户相同，取首行
        cat_ids = batch['propensity_category_ids'][0].to(device)  # [L]
        pop_cat = batch['popularity_category'][0].to(device)
        k_cat = min(100, pop_cat.size(0))
        max_pop_cat = pop_cat.sort(descending=True)[0][:k_cat].sum()
        pop_cat_norm = pop_cat / (max_pop_cat + 1e-8)
        cat_embs = self.category_embeddings(cat_ids)
        cat_embs = F.normalize(cat_embs, p=2, dim=-1)
        logits_cat_all = user_embeddings @ cat_embs.T / self.temperature
        probs_cat_all = torch.softmax(logits_cat_all, dim=1)
        novelty_cat_per_user = (probs_cat_all * pop_cat_norm.unsqueeze(0)).sum(dim=1)
        novelty_category_loss = novelty_cat_per_user.mean()
        # 商品 Novelty loss
        sku_ids = batch['propensity_sku_ids'][0].to(device)
        pop_sku = batch['popularity_sku'][0].to(device)
        k_sku = min(100, pop_sku.size(0))
        max_pop_sku = pop_sku.sort(descending=True)[0][:k_sku].sum()
        pop_sku_norm = pop_sku / (max_pop_sku + 1e-8)
        sku_embs = self.sku_embeddings(sku_ids)
        sku_embs = F.normalize(sku_embs, p=2, dim=-1)
        logits_sku_all = user_embeddings @ sku_embs.T / self.temperature
        probs_sku_all = torch.softmax(logits_sku_all, dim=1)
        novelty_sku_per_user = (probs_sku_all * pop_sku_norm.unsqueeze(0)).sum(dim=1)
        novelty_product_loss = novelty_sku_per_user.mean()
        losses['novelty_category_loss'] = novelty_category_loss
        losses['novelty_product_loss'] = novelty_product_loss
        # 下一个购买商品名称预测损失
        # 使用原始 16 维 quantized vector，不再通过 name_embedding
        pos_name_vec = batch['pos_name'].float()  # (B, name_dim)
        neg_name_vecs = batch['neg_names'].float()  # (B, num_neg, name_dim)
        # L2 归一化标签向量
        pos_name_norm = F.normalize(pos_name_vec, p=2, dim=-1)
        neg_name_norm = F.normalize(neg_name_vecs, p=2, dim=-1)
        # 用户映射到 name 空间
        user_name_pred = self.name_head(user_embeddings)  # (B, name_dim)
        user_name_norm = F.normalize(user_name_pred, p=2, dim=-1)
        # 计算相似度得分
        pos_scores = (user_name_norm * pos_name_norm).sum(-1) / self.temperature  # (B)
        neg_scores = (neg_name_norm * user_name_norm.unsqueeze(1)).sum(-1) / self.temperature  # (B, num_neg)
        logits_name = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=-1)  # (B, 1+num_neg)
        # 交叉熵损失，正样本索引 0
        targets = torch.zeros(batch_size, dtype=torch.long, device=device)
        name_loss = F.cross_entropy(logits_name, targets)
        losses['name_loss'] = name_loss
        weighted_name_loss = self.name_weight * name_loss
        losses['weighted_name_loss'] = weighted_name_loss
        # 将 novelty loss 加入总损失
        novelty_weight = getattr(self.config, 'novelty_weight', 1.0)
        total_loss = weighted_churn + weighted_category + weighted_product + weighted_price + novelty_weight * (novelty_category_loss * catw + novelty_product_loss * pw)
        total_loss = total_loss + weighted_name_loss
        losses['loss'] = total_loss * self.loss_scale
        
        return {
            'user_embedding': user_embeddings,
            'temporal_features': temporal_features,
            'task_outputs': task_outputs,
            **losses
        }

    def clip_gradients(self):
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

UBTModel = UniversalBehavioralTransformer 