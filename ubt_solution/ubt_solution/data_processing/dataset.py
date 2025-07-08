import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import psutil
import gc
import time
from pathlib import Path
from typing import Dict, List
from .target_data import TargetData
from .memory_utils import report_memory_usage
import pandas as pd
from .utils import (
    vectorize_text,
    create_user_chunks,
    load_events_parallel,
    load_product_properties
)
from ..config import Config
import random
from collections import Counter

logger = logging.getLogger(__name__)

class BehaviorSequenceDataset(Dataset):
    def __init__(self, 
                 data_dir: Path,
                 config: Config,
                 mode: str = 'train',
                 test_mode: bool = False):
        self.config = config
        self.mode = mode
        self.test_mode = test_mode
        self.cache_prefix = f"{self.mode}_"
        report_memory_usage("初始化数据集开始")
        
        # 设置数据路径，根据模式选择不同的目录
        if self.mode in ['train', 'valid']:
            # self.base_dir = data_dir
            self.base_dir = data_dir / "input"
        else:
            self.base_dir = data_dir
        
        # 加载相关用户ID
        relevant_clients_path = self.base_dir / "relevant_clients.npy"
        logger.info(f"正在加载相关用户ID，文件路径: {relevant_clients_path}")
        if not relevant_clients_path.exists():
            logger.error(f"文件不存在: {relevant_clients_path}")
            raise FileNotFoundError(f"找不到文件: {relevant_clients_path}")
        self.relevant_clients = np.load(relevant_clients_path)
        relevant_clients1 = np.load("/data_sdc/mhwang/recsys_code/data/target/active_clients.npy")

        if self.mode in ['train', 'valid']:
            combined = np.concatenate((self.relevant_clients, relevant_clients1))
            unique_clients = np.unique(combined)

            # 如果你需要保持原来的变量名
            self.relevant_clients = unique_clients
        else:
            self.relevant_clients = np.load(relevant_clients_path)


        # 如果是测试模式，只取前10000个用户
        if self.test_mode:
            self.relevant_clients = self.relevant_clients[:10000]
            logger.info(f"测试模式：只处理前 {len(self.relevant_clients)} 个用户")
        else:
            logger.info(f"加载了 {len(self.relevant_clients)} 个相关用户")
        
        # 设置用户分块大小和内存调优参数
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        logger.info(f"系统可用内存: {available_memory_gb:.2f} GB")
        
        # 根据可用内存调整分块大小
        if available_memory_gb > 32:  # 如果有大量内存可用
            self.chunk_size = 20000 if self.test_mode else 100000
        elif available_memory_gb > 16:
            self.chunk_size = 10000 if self.test_mode else 50000
        else:  # 内存有限
            self.chunk_size = 5000 if self.test_mode else 25000
            
        logger.info(f"根据系统内存自动设置块大小为: {self.chunk_size}")
        
        self.user_chunks = create_user_chunks(self.relevant_clients, self.chunk_size)
        logger.info(f"将用户分成 {len(self.user_chunks)} 个块进行处理，每块大约 {self.chunk_size} 个用户")
        
        # 加载数据
        self.events = {}
        self.user_sequences = {}
        
        report_memory_usage("开始分块加载事件数据")
        
        # 使用分块策略加载和处理事件
        for chunk_idx, user_chunk in enumerate(self.user_chunks):
            start_time = time.time()
            logger.info(f"处理用户块 {chunk_idx+1}/{len(self.user_chunks)}")
            # 将当前块设置为相关用户集
            self.current_chunk_users = set(user_chunk)
            
            # 加载当前块用户的事件数据并处理序列
            logger.info("开始并行加载当前块的事件数据...")
            chunk_events = load_events_parallel(self.base_dir, self.current_chunk_users)
            
            report_memory_usage(f"块 {chunk_idx+1} 事件数据加载完成")
            
            # 加载商品属性数据（所有块共享）
            if chunk_idx == 0:
                logger.info("开始加载商品属性数据...")
                self.product_properties = load_product_properties(data_dir)
                logger.info("商品属性数据加载完成")
            
            # 构建当前块用户的序列
            logger.info(f"开始构建用户块 {chunk_idx+1} 的序列...")
            chunk_sequences = self._build_user_sequences(chunk_events, user_chunk)
            
            # 合并到总序列字典中
            self.user_sequences.update(chunk_sequences)
            
            # 计算并报告处理时间
            elapsed_time = time.time() - start_time
            logger.info(f"块 {chunk_idx+1} 处理完成，耗时: {elapsed_time:.2f} 秒")
            
            # 释放块事件数据内存并主动触发垃圾回收
            del chunk_events
            del chunk_sequences
            gc.collect()
            
            report_memory_usage(f"块 {chunk_idx+1} 处理完成")
            
        logger.info(f"用户序列构建完成，共 {len(self.user_sequences)} 个用户")
        
        # 加载目标数据
        if self.mode == 'train':
            self.target_data = TargetData(data_dir, target_mode='train')
        elif self.mode == 'valid':
            self.target_data = TargetData(data_dir, target_mode='train')
        else:
            self.target_data = TargetData(data_dir, target_mode='train')
        logger.info(f"目标数据加载完成，目标数据模式: {self.mode}")
        # 用户ID列表和序列长度
        self.client_ids = list(self.user_sequences.keys())
        # 预计算用户在目标数据中的类别和SKU映射
        self.cats_in_target = {}
        self.skus_in_target = {}
        if self.target_data is not None:
            # 选择对应的目标DataFrame
            if self.mode == 'train':
                df = pd.concat([self.target_data.train_df], ignore_index=True)
            elif self.mode == 'valid':
                df = pd.concat([self.target_data.train_df], ignore_index=True)
            else:
                df = pd.concat([self.target_data.train_df], ignore_index=True)
            # 使用 groupby 批量计算 client_id 到 category/sku 列表的映射
            grouped = df.groupby('client_id')
            cat_groups = grouped['category'].unique()
            sku_groups = grouped['sku'].unique()
            # 转换为 int 列表字典
            self.cats_in_target = {cid: arr.astype(int).tolist() for cid, arr in cat_groups.items()}
            self.skus_in_target = {cid: arr.astype(int).tolist() for cid, arr in sku_groups.items()}
            # 对于没有出现的用户，赋予空列表
            for cid in self.client_ids:
                self.cats_in_target.setdefault(cid, [])
                self.skus_in_target.setdefault(cid, [])
        self.max_seq_length = self.config.max_seq_length
        
        # 创建事件类型映射
        self.event_type_map = {
            'product_buy': 0,
            'add_to_cart': 1,
            'remove_from_cart': 2,
            'page_visit': 3,
            'search_query': 4
        }
        
        # 创建商品属性查找字典，加速访问
        logger.info("创建商品属性查找字典...")
        self.product_dict = {}
        try:
            self.product_dict = self.product_properties.set_index('sku').to_dict('index')
            logger.info(f"商品属性字典创建完成，包含 {len(self.product_dict)} 个商品")
        except Exception as e:
            logger.error(f"创建商品属性字典时出错: {str(e)}")
        
        # 初始化样本缓存
        self.item_cache = {}
        # 根据可用内存调整缓存大小
        self.cache_size = min(int(available_memory_gb * 200), 5000)  # 每GB内存缓存约200个样本，但最多5000个
        logger.info(f"设置样本缓存大小为 {self.cache_size}")
        
        # 准备全量候选集用于负采样
        try:
            self.available_categories = set(self.product_properties['category'].astype(int).tolist())
            self.available_categories_list = list(self.available_categories)
            self.available_skus = set(self.product_dict.keys())
            self.available_skus_list = list(self.available_skus)
            # 对可用 SKU 列表进行哈希映射到哈希空间
            self.available_skus_list = [s % self.config.sku_hash_size for s in self.available_skus_list]
            # 更新可用 SKU 集合
            self.available_skus = set(self.available_skus_list)
            logger.info(f"哈希后可用 SKU 数量: {len(self.available_skus_list)}, 可用类别数量: {len(self.available_categories_list)}")
        except Exception as e:
            logger.warning(f"初始化全量候选集失败: {e}")
            self.available_categories_list = []
            self.available_skus_list = []
        
        # 负采样数量
        self.num_negative_samples = getattr(self.config, 'num_negative_samples', 10)
        
        # 预计算用户交互过的 SKU 和类别集合，用于负采样
        self.interacted_skus = {}
        self.interacted_categories = {}
        for cid, seq in self.user_sequences.items():
            skus_set = {evt['sku'] for evt in seq if evt.get('sku', -1) != -1}
            self.interacted_skus[cid] = skus_set
            cat_set = set()
            for sku in skus_set:
                if sku in self.product_dict:
                    cat_set.add(int(self.product_dict[sku].get('category', 0)))
            self.interacted_categories[cid] = cat_set
        
        # 预计算用户全局特征
        self.global_feats = {}
        for cid, seq in self.user_sequences.items():
            types = [evt['event_type'] for evt in seq]
            counts = Counter(types)
            total = len(seq) if len(seq) > 0 else 1
            # 事件类型占比
            ratio_buy = counts.get('product_buy', 0) / total
            ratio_add = counts.get('add_to_cart', 0) / total
            ratio_remove = counts.get('remove_from_cart', 0) / total
            ratio_visit = counts.get('page_visit', 0) / total
            ratio_query = counts.get('search_query', 0) / total
            # SKU事件总数
            sku_events = (counts.get('product_buy', 0) + counts.get('add_to_cart', 0) + counts.get('remove_from_cart', 0))
            sku_events = sku_events if sku_events > 0 else 1
            unique_sku_ratio = len(self.interacted_skus.get(cid, set())) / sku_events
            unique_cat_ratio = len(self.interacted_categories.get(cid, set())) / sku_events
            # 价格统计
            prices = [
                self.product_dict[s]['price'] for s in self.interacted_skus.get(cid, set())
                if s in self.product_dict
            ]
            if prices:
                price_mean = float(np.mean(prices)) / self.config.max_price
                price_std = float(np.std(prices)) / self.config.max_price
            else:
                price_mean, price_std = 0.0, 0.0
            # 时间统计
            times = [evt['timestamp'] for evt in seq]
            if len(times) > 1:
                times_arr = np.array(times)
                duration = float(times_arr[-1] - times_arr[0])
                median_delta = float(np.median(np.diff(times_arr)))
            else:
                duration, median_delta = 0.0, 0.0
            duration_norm = duration / self.config.max_seq_length
            median_delta_norm = median_delta / self.config.max_seq_length
            feats = [
                ratio_buy, ratio_add, ratio_remove, ratio_visit, ratio_query,
                unique_sku_ratio, unique_cat_ratio,
                price_mean, price_std,
                duration_norm, median_delta_norm
            ]
            self.global_feats[cid] = feats
        
        # 添加：保存候选 category 和 sku 及其流行度，用于 novelty loss
        import torch
        self.propensity_category_ids = torch.from_numpy(self.target_data.propensity_category).long()
        self.popularity_category        = torch.from_numpy(self.target_data.popularity_category).float()
        self.propensity_sku_ids         = torch.from_numpy(self.target_data.propensity_sku).long() % self.config.sku_hash_size
        self.popularity_sku             = torch.from_numpy(self.target_data.popularity_sku).float()
        
        # cache路径加mode前缀
      
        
        report_memory_usage("数据集初始化完成")
        
    def _build_user_sequences(self, events: Dict[str, List], user_chunk: List[int]) -> Dict[int, List[Dict]]:
        """构建用户序列"""
        logger.info("开始构建用户序列...")
        report_memory_usage("开始构建用户序列")
        user_chunk_set = set(user_chunk)
        
        # 检查缓存文件是否存在
        cache_dir = Path(__file__).parent / "cache"
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"{self.cache_prefix}user_sequences_{hash(tuple(sorted(user_chunk_set)))}.pkl"
        
        if cache_file.exists():
            logger.info(f"找到缓存文件 {cache_file}，尝试加载...")
            try:
                import pickle
                with open(cache_file, 'rb') as f:
                    user_sequences = pickle.load(f)
                logger.info(f"成功从缓存加载 {len(user_sequences)} 个用户序列")
                return user_sequences
            except Exception as e:
                logger.warning(f"加载缓存文件失败: {str(e)}，将重新构建序列")
        
        # 合并所有事件到一个DataFrame
        logger.info("合并事件数据...")
        try:
            # 预先估计大小，避免多次内存分配
            total_rows = sum(len(df) for df in events.values())
            logger.info(f"预计合并 {total_rows} 行事件数据")
            
            # 只合并有数据的DataFrame
            non_empty_events = [df for df in events.values() if not df.empty]
            if not non_empty_events:
                logger.warning("没有事件数据，将返回空序列")
                return {client_id: self._create_default_sequence() for client_id in user_chunk_set}
            
            all_events = pd.concat(non_empty_events, ignore_index=True)
        except Exception as e:
            logger.error(f"合并事件数据时发生错误: {str(e)}")
            # 创建一个空DataFrame
            all_events = pd.DataFrame(columns=['client_id', 'timestamp', 'event_type', 'sku'])
        
        if len(all_events) == 0:
            logger.warning("没有事件数据，将返回空序列")
            return {client_id: self._create_default_sequence() for client_id in user_chunk_set}
        
        # 优化内存使用
        logger.info("优化数据类型以减少内存占用...")
        # 将整数列转换为最小可能的类型
        if 'client_id' in all_events.columns:
            all_events['client_id'] = pd.to_numeric(all_events['client_id'], downcast='integer')
        if 'sku' in all_events.columns:
            all_events['sku'] = pd.to_numeric(all_events['sku'], downcast='integer')
        
        # 转换时间戳
        logger.info("处理时间戳...")
        all_events['timestamp'] = pd.to_datetime(all_events['timestamp'], errors='coerce')
        all_events = all_events.dropna(subset=['timestamp'])
        
        report_memory_usage("事件数据合并和类型优化完成")
        
        # 使用分组优化处理逻辑
        logger.info("按用户分组预处理事件...")
        
        # 先找出有事件的用户
        users_with_events = all_events['client_id'].unique()
        logger.info(f"在事件数据中发现 {len(users_with_events)} 个用户")
        
        # 为没有事件的用户创建默认序列
        user_sequences = {client_id: self._create_default_sequence() 
                         for client_id in user_chunk_set if client_id not in users_with_events}
        
        # 分批处理以减少内存消耗
        max_sequence_length = self.config.max_seq_length
        
        # 根据事件数据大小动态调整批次大小
        events_per_user = len(all_events) / max(1, len(users_with_events))
        batch_size = 5000  # 增大批次大小
            
        logger.info(f"每用户平均事件数: {events_per_user:.1f}, 设置批次大小: {batch_size}")
        
        # 计算总批次数
        num_batches = (len(users_with_events) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(users_with_events))
            batch_users = users_with_events[batch_start:batch_end]
            
            logger.info(f"处理用户批次 {batch_idx+1}/{num_batches}，包含 {len(batch_users)} 个用户")
            
            # 只选择当前批次用户的事件
            batch_events = all_events[all_events['client_id'].isin(batch_users)].copy()
            
            # 按用户分组
            user_grouped_events = batch_events.groupby('client_id')
            
            # 高效处理每个用户的序列
            batch_sequences = {}
            for client_id, user_df in user_grouped_events:
                try:
                    # 按时间排序
                    user_events = user_df.sort_values('timestamp')
                    
                    # # 对 page_visit 事件只保留最近100次
                    mask = user_events['event_type'] == 'page_visit'
                    pv_events = user_events[mask]
                    if len(pv_events) > 300:
                        pv_events = pv_events.iloc[-300:]
                    other_events = user_events[~mask]
                    user_events = pd.concat([other_events, pv_events]).sort_values('timestamp')
                    
                    # 如果序列太长，只保留最近的记录
                    if len(user_events) > max_sequence_length:
                        user_events = user_events.iloc[-max_sequence_length:]
                    
                    # 标准化时间戳
                    min_ts = user_events['timestamp'].min()
                    # 使用向量化操作，不是循环
                    user_events['norm_timestamp'] = (user_events['timestamp'] - min_ts).dt.total_seconds() / 86400.0
                    
                    # 使用列表推导式构建序列 - 比循环更快
                    sequence = [
                        {
                            'event_type': row['event_type'],
                            'timestamp': row['norm_timestamp'],
                            'sku': row['sku'],
                            'url': row.get('url', -1) if row['event_type'] == 'page_visit' and 'url' in row else -1,
                            **({'query': row['query']} if row['event_type'] == 'search_query' and 'query' in row else {})
                        }
                        for _, row in user_events.iterrows()
                    ]
                    
                    batch_sequences[client_id] = sequence
                except Exception as e:
                    logger.error(f"处理用户 {client_id} 序列时发生错误: {str(e)}")
                    # 出错时创建默认序列
                    batch_sequences[client_id] = self._create_default_sequence()
            
            # 更新总序列字典
            user_sequences.update(batch_sequences)
            
            # 清理批次数据以释放内存
            del batch_events, user_grouped_events, batch_sequences
            gc.collect()
            
            # 每4个批次报告一次内存使用情况
            if batch_idx % 4 == 0:
                report_memory_usage(f"用户批次 {batch_idx+1}/{num_batches} 处理完成")
        
        # 保存到缓存文件
        try:
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(user_sequences, f)
            logger.info(f"成功将 {len(user_sequences)} 个用户序列保存到缓存文件 {cache_file}")
        except Exception as e:
            logger.warning(f"保存缓存文件失败: {str(e)}")
        
        report_memory_usage("用户序列构建完成")
        logger.info(f"成功构建了 {len(user_sequences)} 个用户序列")
        return user_sequences
    
    def _create_default_sequence(self) -> List[Dict]:
        """为空序列创建默认值"""
        return [{'event_type': 'page_visit', 'timestamp': 0.0}]
    
    def __len__(self) -> int:
        return len(self.client_ids)
    
    def __getitem__(self, idx):
        # 首先检查缓存
        if idx in self.item_cache:
            return self.item_cache[idx]
        
        client_id = self.client_ids[idx]
        sequence = self.user_sequences[client_id]
        
        # 用户全局特征
        user_feats = torch.tensor(self.global_feats.get(client_id, [0]*11), dtype=torch.float)
        
        # 预分配内存
        event_types = torch.zeros(self.max_seq_length, dtype=torch.long)
        timestamps = torch.zeros(self.max_seq_length, dtype=torch.float)
        categories = torch.zeros(self.max_seq_length, dtype=torch.long)
        prices = torch.zeros(self.max_seq_length, dtype=torch.long)
        names = torch.zeros((self.max_seq_length, 16), dtype=torch.float)
        queries = torch.zeros((self.max_seq_length, 16), dtype=torch.float)
        mask = torch.zeros(self.max_seq_length, dtype=torch.bool)
        
        # 新增：URL和Item ID特征
        urls = torch.zeros(self.max_seq_length, dtype=torch.long)
        item_ids = torch.zeros(self.max_seq_length, dtype=torch.long)
        
        # 填充序列 - 使用向量化操作替代循环
        seq_len = min(len(sequence), self.max_seq_length)
        
        # 批量处理事件类型
        for i in range(seq_len):
            event = sequence[i]
            event_type = event['event_type']
            event_code = self.event_type_map.get(event_type, 0)
            
            event_types[i] = event_code
            timestamps[i] = float(event['timestamp'])
            mask[i] = True
            
            # 处理商品特征
            sku = event.get('sku', -1)
            if sku != -1:
                # 对序列中的SKU进行哈希映射
                item_ids[i] = sku % self.config.sku_hash_size
                
                if event_code in [self.event_type_map['product_buy'], 
                                  self.event_type_map['add_to_cart'], 
                                  self.event_type_map['remove_from_cart']] and sku != -1:
                    if sku in self.product_dict:
                        product = self.product_dict[sku]
                        categories[i] = int(product.get('category', 0))
                        prices[i] = min(100, int(product.get('price', 0)))
                        names[i] = torch.tensor(vectorize_text(str(product.get('name', ''))))
            
            # 处理搜索查询
            if event_type == 'search_query' and 'query' in event:
                queries[i] = torch.tensor(vectorize_text(str(event['query'])))
                
            # 处理URL
            if event_type == 'page_visit' and 'url' in event:
                urls[i] = int(event['url'])
        
        # 获取目标值
        if self.target_data is not None:
            try:
                category_propensity = self.target_data.get_category_propensity_target(client_id)
                category_tensor = torch.tensor(category_propensity, dtype=torch.float)
            except Exception as e:
                logger.warning(f"获取用户 {client_id} 的品类倾向目标时出错: {str(e)}")
                category_tensor = torch.zeros(len(self.target_data.propensity_category), dtype=torch.float)
            try:
                product_propensity = self.target_data.get_product_propensity_target(client_id)
                product_tensor = torch.tensor(product_propensity, dtype=torch.float)
            except Exception as e:
                logger.warning(f"获取用户 {client_id} 的商品倾向目标时出错: {str(e)}")
                product_tensor = torch.zeros(len(self.target_data.propensity_sku), dtype=torch.float)
            try:
                churn_target = self.target_data.get_churn_target(client_id)
                churn_tensor = torch.tensor(churn_target, dtype=torch.float)
            except Exception as e:
                logger.warning(f"获取用户 {client_id} 的流失目标时出错: {str(e)}")
                churn_tensor = torch.tensor(0.0, dtype=torch.float)
        else:
            category_tensor = torch.zeros(100, dtype=torch.float)  # 100为默认类别数
            product_tensor = torch.zeros(100, dtype=torch.float)
            churn_tensor = torch.tensor(0.0, dtype=torch.float)
        
        # 检查并替换NaN
        if torch.isnan(category_tensor).any():
            category_tensor = torch.zeros_like(category_tensor)
        if torch.isnan(product_tensor).any():
            product_tensor = torch.zeros_like(product_tensor)
        
        # 获取预计算的用户交互类别和SKU
        cats = self.cats_in_target.get(client_id, [])
        skus = self.skus_in_target.get(client_id, [])
        # 若目标列表为空，则使用序列中最近一次product_buy事件作为正样本
        if not cats or not skus:
            for evt in reversed(sequence):
                if evt.get('event_type') == 'product_buy':
                    buy_sku = evt.get('sku', None)
                    if buy_sku is not None and buy_sku in self.product_dict:
                        buy_cat = int(self.product_dict[buy_sku].get('category', 0))
                        if not cats:
                            cats = [buy_cat]
                        if not skus:
                            skus = [buy_sku]
                        break
        if not cats or not skus:
            for evt in reversed(sequence):
                if evt.get('event_type') == 'add_to_cart':
                    buy_sku = evt.get('sku', None)
                    if buy_sku is not None and buy_sku in self.product_dict:
                        buy_cat = int(self.product_dict[buy_sku].get('category', 0))
                        if not cats:
                            cats = [buy_cat]
                        if not skus:
                            skus = [buy_sku]
                        break
        if not cats or not skus:
            for evt in reversed(sequence):
                if evt.get('event_type') == 'remove_from_cart':
                    buy_sku = evt.get('sku', None)
                    if buy_sku is not None and buy_sku in self.product_dict:
                        buy_cat = int(self.product_dict[buy_sku].get('category', 0))
                        if not cats:
                            cats = [buy_cat]
                        if not skus:
                            skus = [buy_sku]
                        break
        # 正样本ID
        pos_cat_id = cats[0] if len(cats) > 0 else 0
        pos_sku_id = skus[0] if len(skus) > 0 else 0
        # 对正样本SKU进行哈希映射
        pos_sku_id = pos_sku_id % self.config.sku_hash_size
        
        # 负采样：类别，排除已交互和目标，并加入 target_data 的 propensity_category
        interacted_cat = self.interacted_categories.get(client_id, set())
        pos_cat_set = set(cats)
        neg_cat_ids = []
        if self.target_data is not None:
            for cat in self.target_data.propensity_category.tolist():
                if cat not in interacted_cat and cat not in pos_cat_set:
                    neg_cat_ids.append(cat)
                    if len(neg_cat_ids) >= self.num_negative_samples:
                        break
        while len(neg_cat_ids) < self.num_negative_samples and self.available_categories_list:
            cand = random.choice(self.available_categories_list)
            if cand not in interacted_cat and cand not in pos_cat_set and cand not in neg_cat_ids:
                neg_cat_ids.append(cand)
        neg_cat_ids = torch.tensor(neg_cat_ids, dtype=torch.long)

        # 负采样：商品，排除已交互和目标，并加入 target_data 的 propensity_sku
        interacted_sku = self.interacted_skus.get(client_id, set())
        pos_sku_set = set(skus)
        neg_sku_ids = []
        if self.target_data is not None:
            for sku in self.target_data.propensity_sku.tolist():
                if sku not in interacted_sku and sku not in pos_sku_set:
                    neg_sku_ids.append(sku)
                    if len(neg_sku_ids) >= self.num_negative_samples:
                        break
        while len(neg_sku_ids) < self.num_negative_samples and self.available_skus_list:
            cand = random.choice(self.available_skus_list)
            if cand not in interacted_sku and cand not in pos_sku_set and cand not in neg_sku_ids:
                neg_sku_ids.append(cand)
        neg_sku_ids = torch.tensor(neg_sku_ids, dtype=torch.long)
        
        # 对负样本SKU进行哈希映射
        neg_sku_ids = neg_sku_ids % self.config.sku_hash_size
        
        # 对skus_in_target列表进行哈希映射
        hashed_skus = [s % self.config.sku_hash_size for s in skus]
        
        # 构建结果
        result = {
            'event_types': event_types,
            'timestamps': timestamps,
            'categories': categories,
            'prices': prices,
            'names': names,
            'queries': queries,
            'mask': mask,
            'client_id': torch.tensor(client_id),
            'churn': churn_tensor,
            'category_propensity': category_tensor,
            'product_propensity': product_tensor,
            'urls': urls,
            'item_ids': item_ids,
            'cats_in_target': cats,
            'skus_in_target': hashed_skus,
            'pos_cat_id': torch.tensor(pos_cat_id, dtype=torch.long),
            'neg_cat_ids': neg_cat_ids,
            'pos_sku_id': torch.tensor(pos_sku_id, dtype=torch.long),
            'neg_sku_ids': neg_sku_ids,
            'user_feats': user_feats,
        }
        # 添加：将 novelty 计算所需数据加入 batch
        result['propensity_category_ids'] = self.propensity_category_ids
        result['popularity_category']      = self.popularity_category
        result['propensity_sku_ids']      = self.propensity_sku_ids
        result['popularity_sku']           = self.popularity_sku
        
        # 新增：price回归目标，计算平均价格及是否有目标
        if skus:
            prices_list = [self.product_dict.get(s, {}).get('price', 0) for s in skus]
            avg_price = sum(prices_list) / len(prices_list)
            has_price = 1.0
        else:
            avg_price = 0.0
            has_price = 0.0
        result['price_target'] = torch.tensor(avg_price, dtype=torch.float)
        result['has_price_target'] = torch.tensor(has_price, dtype=torch.float)
        
        # 新增：下一个购买商品name预测目标及负样本采样
        # 正样本名称向量（量化后的16维ID）
        if skus:
            name_vec = torch.tensor(vectorize_text(str(self.product_dict[skus[0]].get('name', ''))), dtype=torch.long)
        else:
            name_vec = torch.zeros(16, dtype=torch.long)
        # 负样本名称向量列表
        neg_name_vecs = []
        for neg_sku in neg_sku_ids.tolist():
            neg_name = str(self.product_dict.get(int(neg_sku), {}).get('name', ''))
            neg_name_vecs.append(torch.tensor(vectorize_text(neg_name), dtype=torch.long))
        neg_name_vecs = torch.stack(neg_name_vecs, dim=0)
        result['pos_name'] = name_vec
        result['neg_names'] = neg_name_vecs
        
        # 缓存结果，如果缓存过大则移除最早项
        if len(self.item_cache) >= self.cache_size:
            # 简单策略：随机移除一个项
            remove_key = list(self.item_cache.keys())[0]
            del self.item_cache[remove_key]
        
        self.item_cache[idx] = result
        # 直接输出具体值，便于观察
        # print("返回结果:", {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in result.items()})
        return result 