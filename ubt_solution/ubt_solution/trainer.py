import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
from .config import Config
from .model import UniversalBehavioralTransformer
from .data_processor import create_data_loaders
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC
from torchmetrics import MeanSquaredError
import random

logger = logging.getLogger(__name__)

class UBTTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # 检查 CUDA 可用性
        self.use_cuda = torch.cuda.is_available() and self.config.device.startswith("cuda")
        
        # 设置设备
        self.device = torch.device(self.config.device)
        
        # 初始化优化器 - 使用较低的学习率以提高稳定性
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate * 0.5,  # 降低学习率
            weight_decay=config.weight_decay,
            eps=1e-8  # 提高数值稳定性
        )
        
        # 使用更稳定的学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,  # 增加耐心
            verbose=True,
            min_lr=1e-6  # 设置最小学习率
        )
            
        # 将模型移动到指定设备
        self.model = self.model.to(self.device)
        
        # 初始化评估指标：流失AUC 与 价格回归 MSE
        self.metrics = {
            'churn': BinaryAUROC(),
            'price_mse': MeanSquaredError(),
        }
        
        # 将指标移动到设备
        for metric in self.metrics.values():
            metric.to(self.device)
        
        # 训练状态跟踪
        self.epochs_without_improvement = {
            'churn_loss': 0,
            'category_loss': 0,
            'product_loss': 0
        }
        
        # 记录最佳损失
        self.best_losses = {
            'churn_loss': float('inf'),
            'category_loss': float('inf'),
            'product_loss': float('inf')
        }
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        task_losses = {
            'churn_loss': 0.0,
            'category_loss': 0.0,
            'product_loss': 0.0,
            'price_loss': 0.0,
            'novelty_category_loss': 0.0,
            'novelty_product_loss': 0.0,
        }
        total_samples = 0
        skipped_batches = 0
        
        ds = train_loader.dataset
        available_cats = ds.available_categories_list
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            logger.info(f"Batch {batch_idx}: 开始训练")
            try:
                # 将数据移动到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                # 动态负采样 - 每个epoch重新采样 neg_cat_ids 和 neg_sku_ids
                num_neg = self.config.num_negative_samples
                # 类别负采样
                prop_cats = batch.get('propensity_category_ids', [])[0].tolist() if 'propensity_category_ids' in batch else []
                new_neg_cats = []
                batch_size = batch['client_id'].size(0)
                for i in range(batch_size):
                    pos = batch['cats_in_target'][i]
                    negs = []
                    # 首先从倾向性列表采样
                    for cat in prop_cats:
                        if cat not in pos and len(negs) < num_neg:
                            negs.append(cat)
                    # 随机补采样
                    while len(negs) < num_neg and available_cats:
                        c = random.choice(available_cats)
                        if c not in pos and c not in negs:
                            negs.append(c)
                    new_neg_cats.append(torch.tensor(negs, dtype=torch.long, device=self.device))
                batch['neg_cat_ids'] = torch.stack(new_neg_cats, dim=0)
                # 商品负采样
                # 使用哈希后的可用 SKU 列表进行采样，防止索引超出范围
                available_skus = [s % self.config.sku_hash_size for s in ds.available_skus_list]
                prop_skus = batch.get('propensity_sku_ids', [])[0].tolist() if 'propensity_sku_ids' in batch else []
                new_neg_skus = []
                for i in range(batch_size):
                    pos = batch['skus_in_target'][i]
                    negs = []
                    for sku in prop_skus:
                        if sku not in pos and len(negs) < num_neg:
                            negs.append(sku)
                    while len(negs) < num_neg and available_skus:
                        s = random.choice(available_skus)
                        if s not in pos and s not in negs:
                            negs.append(s)
                    # 对采样到的 SKU 做哈希映射
                    hashed_negs = [nid % self.config.sku_hash_size for nid in negs]
                    new_neg_skus.append(torch.tensor(hashed_negs, dtype=torch.long, device=self.device))
                batch['neg_sku_ids'] = torch.stack(new_neg_skus, dim=0)
                
                # 检查输入数据是否有NaN
                has_nan = False
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                        logger.warning(f"NaN detected in input tensor {k}, skipping batch")
                        has_nan = True
                        break
                
                if has_nan:
                    skipped_batches += 1
                    continue
                
                # 统计样本数
                # batch_size 已在负采样前获取
                total_samples += batch_size
                
                # 清零梯度
                self.optimizer.zero_grad()
                
                # 前向传播 - 简化，不使用混合精度
                outputs = self.model(batch)
                
                loss = outputs['loss']
                # 每个batch输出三个任务的损失
                logger.info(f"batch {batch_idx} losses: churn_loss={outputs['churn_loss'].item():.4f}, category_loss={outputs['category_loss'].item():.4f}, product_loss={outputs['product_loss'].item():.4f}, novelty_category_loss={outputs['novelty_category_loss'].item():.4f}, novelty_product_loss={outputs['novelty_product_loss'].item():.4f}, price_loss={outputs['price_loss'].item():.4f}, total_loss={loss.item():.4f}")
                
                # 记录各任务损失
                for task in task_losses.keys():
                    if task in outputs:
                        task_losses[task] += outputs[task].item() * batch_size
                
                # 检查损失是否为NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN or Inf loss detected in batch {batch_idx}, skipping")
                else:
                    # 反向传播
                    loss.backward()

                    # 梯度裁剪，防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    # 检查梯度是否有无限值或NaN，并将其置为零
                    valid_gradients = True
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                logger.warning(f"Invalid gradients detected in {name}, setting to zero")
                                param.grad = torch.zeros_like(param.grad) # 将无效梯度置为零

                    # 更新参数
                    self.optimizer.step()
                    
                    total_loss += loss.item() * len(batch['client_id'])
                
            except RuntimeError as e:
                logger.error(f"RuntimeError in batch {batch_idx}: {str(e)}")
                skipped_batches += 1
                continue
        
        if skipped_batches > 0:
            logger.warning(f"Skipped {skipped_batches} batches during training")
        logger.info(f"total_loss: {total_loss}, total_samples: {total_samples}")
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        logger.info(f"Epoch average loss: {avg_loss:.4f}")
        
        # 返回损失字典
        avg_task_losses = {}
        for task, loss_sum in task_losses.items():
            avg_task_losses[task] = loss_sum / total_samples if total_samples > 0 else float('inf')
            logger.info(f"Epoch average {task}: {avg_task_losses[task]:.4f}")
        
        return avg_loss, avg_task_losses
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        task_losses = {
            'churn_loss': 0.0,
            'category_loss': 0.0,
            'product_loss': 0.0,
            'price_loss': 0.0,
            'novelty_category_loss': 0.0,
            'novelty_product_loss': 0.0,
        }
        total_samples = 0
        
        # 收集预测/标签，用于计算指标
        task_metrics = {
            'churn': {'preds': [], 'targets': []},
            'price': {'preds': [], 'targets': []}
        }
        
        # 排序指标收集
        category_recalls = []
        category_ndcgs = []
        product_recalls = []
        product_ndcgs = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # 将数据移到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # 检查输入数据是否有NaN
                has_nan = False
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                        has_nan = True
                        break
                
                if has_nan:
                    continue
                
                # 前向传播
                outputs = self.model(batch)
                
                # 检查损失是否为NaN
                if torch.isnan(outputs['loss']) or torch.isinf(outputs['loss']):
                    continue
                
                # 计算损失
                loss = outputs['loss']
                total_loss += loss.item() * batch['client_id'].size(0)
                total_samples += batch['client_id'].size(0)
                
                # 记录各任务损失
                for task in ['churn_loss','category_loss','product_loss','price_loss','novelty_category_loss','novelty_product_loss']:
                    if task in outputs:
                        task_losses[task] += outputs[task].item() * batch['client_id'].size(0)
                
                # 仅收集 churn 预测用于 AUC
                preds = outputs['task_outputs'].get('churn', None)
                if preds is not None:
                    preds = torch.sigmoid(torch.clamp(preds, min=-10, max=10))
                    task_metrics['churn']['preds'].append(preds.detach().cpu())
                    task_metrics['churn']['targets'].append(batch['churn'].detach().cpu())
                # 计算类别和商品的排序指标
                user_embs = outputs['user_embedding']  # (B, D)
                user_embs = F.normalize(user_embs, p=2, dim=-1)
                batch_size = user_embs.size(0)
                for i in range(batch_size):
                    # 类别排序
                    pos_ids = batch['cats_in_target'][i]
                    neg_ids = batch['neg_cat_ids'][i]
                    if isinstance(pos_ids, list) and len(pos_ids) > 0:
                        pos_tensor = torch.tensor(pos_ids, dtype=torch.long, device=self.device)
                        sampled_ids = torch.cat([pos_tensor, neg_ids.to(self.device)], dim=0)
                        emb = self.model.category_embeddings(sampled_ids)
                        emb = F.normalize(emb, p=2, dim=-1)
                        scores = (emb * user_embs[i].unsqueeze(0)).sum(dim=1)
                        P = pos_tensor.size(0)
                        K = neg_ids.size(0)
                        relevance = torch.cat([torch.ones(P, device=self.device), torch.zeros(K, device=self.device)], dim=0)
                        _, idx_sorted = scores.sort(descending=True)
                        rel_sorted = relevance[idx_sorted]
                        topk = min(20, rel_sorted.size(0))
                        rel_topk = rel_sorted[:topk]
                        # recall@20
                        rec = rel_topk.sum().item() / P
                        category_recalls.append(rec)
                        # ndcg@20
                        idx_arr = torch.arange(1, topk+1, device=self.device, dtype=torch.float)
                        discount = 1.0 / torch.log2(idx_arr + 1)
                        dcg = (rel_topk * discount).sum().item()
                        ideal_rel = torch.ones(min(P, topk), device=self.device, dtype=torch.float)
                        ideal_discount = discount[:ideal_rel.size(0)]
                        idcg = (ideal_rel * ideal_discount).sum().item()
                        ndcg = dcg / idcg if idcg > 0 else 0.0
                        category_ndcgs.append(ndcg)
                    # 商品排序
                    pos_ids = batch['skus_in_target'][i]
                    neg_ids = batch['neg_sku_ids'][i]
                    if isinstance(pos_ids, list) and len(pos_ids) > 0:
                        pos_tensor = torch.tensor(pos_ids, dtype=torch.long, device=self.device)
                        sampled_ids = torch.cat([pos_tensor, neg_ids.to(self.device)], dim=0)
                        emb = self.model.sku_embeddings(sampled_ids)
                        emb = F.normalize(emb, p=2, dim=-1)
                        scores = (emb * user_embs[i].unsqueeze(0)).sum(dim=1)
                        P = pos_tensor.size(0)
                        K = neg_ids.size(0)
                        relevance = torch.cat([torch.ones(P, device=self.device), torch.zeros(K, device=self.device)], dim=0)
                        _, idx_sorted = scores.sort(descending=True)
                        rel_sorted = relevance[idx_sorted]
                        topk = min(20, rel_sorted.size(0))
                        rel_topk = rel_sorted[:topk]
                        # recall@20
                        rec = rel_topk.sum().item() / P
                        product_recalls.append(rec)
                        # ndcg@20
                        idx_arr = torch.arange(1, topk+1, device=self.device, dtype=torch.float)
                        discount = 1.0 / torch.log2(idx_arr + 1)
                        dcg = (rel_topk * discount).sum().item()
                        ideal_rel = torch.ones(min(P, topk), device=self.device, dtype=torch.float)
                        ideal_discount = discount[:ideal_rel.size(0)]
                        idcg = (ideal_rel * ideal_discount).sum().item()
                        ndcg = dcg / idcg if idcg > 0 else 0.0
                        product_ndcgs.append(ndcg)
                
                # 收集 price 回归预测/标签（仅有标签样本）
                price_pred = outputs['task_outputs'].get('price', None)
                price_target = batch.get('price_target', None)
                has_mask = batch.get('has_price_target', None)
                if price_pred is not None and price_target is not None and has_mask is not None:
                    mask = has_mask > 0.5
                    if mask.any():
                        task_metrics['price']['preds'].append(price_pred[mask].detach().cpu())
                        task_metrics['price']['targets'].append(price_target[mask].detach().cpu())
        
        # 计算指标
        metrics = {'val_loss': total_loss / total_samples if total_samples > 0 else float('inf')}
        
        # 添加任务损失到指标（price_loss按有效样本数平均）
        metrics['churn_loss'] = task_losses['churn_loss']/total_samples if total_samples>0 else float('inf')
        metrics['category_loss'] = task_losses['category_loss']/total_samples if total_samples>0 else float('inf')
        metrics['product_loss'] = task_losses['product_loss']/total_samples if total_samples>0 else float('inf')
        metrics['price_loss'] = task_losses['price_loss']/total_samples if total_samples>0 else float('inf')
        
        # churn AUC
        if task_metrics['churn']['preds']:
            preds = torch.cat(task_metrics['churn']['preds'])
            targets = torch.cat(task_metrics['churn']['targets'])
            metrics['churn_auc'] = self.metrics['churn'](preds, targets).item()
        # price 回归指标：MSE
        if task_metrics['price']['preds']:
            preds = torch.cat(task_metrics['price']['preds'])
            targets = torch.cat(task_metrics['price']['targets'])
            metrics['price_mse'] = self.metrics['price_mse'](preds, targets).item()
        
        # 添加排序指标
        metrics['category_recall@20'] = sum(category_recalls)/len(category_recalls) if category_recalls else 0.0
        metrics['category_ndcg@20'] = sum(category_ndcgs)/len(category_ndcgs) if category_ndcgs else 0.0
        metrics['product_recall@20'] = sum(product_recalls)/len(product_recalls) if product_recalls else 0.0
        metrics['product_ndcg@20'] = sum(product_ndcgs)/len(product_ndcgs) if product_ndcgs else 0.0
        
        return metrics
    
    def train(self, train_loader: Optional[DataLoader] = None, val_loader: Optional[DataLoader] = None):
        """训练模型"""
        best_val_loss = float('inf')
        patience_counter = 0
        # 记录最佳指标，churn_auc 越高越好，price_mse 越低越好
        best_task_metrics = {
            'churn_auc': 0.0,
            'price_mse': float('inf')
        }
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            # 训练一个epoch
            train_loss, train_task_losses = self.train_epoch(train_loader)
            
            # 验证
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics['val_loss']
                
                # 检查各任务损失是否改善
                for task, loss in val_metrics.items():
                    if task in self.best_losses:
                        if loss < self.best_losses[task]:
                            self.best_losses[task] = loss
                            self.epochs_without_improvement[task] = 0
                            logger.info(f"{task} improved: {loss:.4f}")
                        else:
                            self.epochs_without_improvement[task] += 1
                            logger.info(f"{task} no improvement for {self.epochs_without_improvement[task]} epochs")
                
                # 记录最佳任务指标
                for metric_name in best_task_metrics:
                    if metric_name in val_metrics:
                        if metric_name == 'price_mse':
                            # 回归指标越低越好
                            if val_metrics[metric_name] < best_task_metrics[metric_name]:
                                best_task_metrics[metric_name] = val_metrics[metric_name]
                                logger.info(f"New best {metric_name}: {best_task_metrics[metric_name]:.4f}")
                        else:
                            # 分类指标越高越好
                            if val_metrics[metric_name] > best_task_metrics[metric_name]:
                                best_task_metrics[metric_name] = val_metrics[metric_name]
                                logger.info(f"New best {metric_name}: {best_task_metrics[metric_name]:.4f}")
                
                # 更新学习率
                self.scheduler.step(val_loss)
                
                # 早停检查 - 基于总体损失
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最佳模型
                    torch.save(self.model.state_dict(), 'best_model.pt')
                    logger.info(f"保存最佳模型，验证损失: {val_loss:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        logger.info("Early stopping triggered")
                        break
                
                # 检查特定任务是否没有改善,这里去了max 5!!!
                if all(self.epochs_without_improvement[task] >= max(5, self.config.patience // 2) for task in ['category_loss', 'product_loss']):
                    logger.info("品类和产品倾向性任务长时间没有改善，提前停止训练")
                    break
                
                # 记录指标
                logger.info(f"Validation metrics: {val_metrics}")
    
    def generate_embeddings(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """生成用户表示"""
        self.model.eval()
        embeddings = []
        client_ids = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Generating embeddings"):
                # 将数据移到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # 检查输入数据是否有NaN
                has_nan = False
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                        has_nan = True
                        break
                
                if has_nan:
                    continue
                
                # 前向传播
                try:
                    outputs = self.model(batch)
                    
                    # 检查输出是否有NaN
                    if torch.isnan(outputs['user_embedding']).any():
                        continue
                    
                    # 收集用户表示和ID，将 embedding 转 float16
                    emb_fp16 = outputs['user_embedding'].half().cpu().numpy()
                    embeddings.append(emb_fp16)
                    client_ids.append(batch['client_id'].cpu().numpy())
                except Exception as e:
                    logger.error(f"Error generating embeddings: {str(e)}")
                    continue
        
        # 合并结果
        if embeddings and client_ids:
            embeddings = np.concatenate(embeddings, axis=0)
            client_ids = np.concatenate(client_ids, axis=0)
            return client_ids, embeddings
        else:
            return np.array([]), np.array([])
