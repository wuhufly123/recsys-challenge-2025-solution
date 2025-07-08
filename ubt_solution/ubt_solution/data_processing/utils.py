import numpy as np
import pandas as pd
import logging
from typing import Dict, List
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

def vectorize_text(text: str, max_length: int = 16) -> np.ndarray:
    """将文本转换为固定长度的向量"""
    vector = np.zeros(max_length, dtype=np.float32)
    if isinstance(text, str):
        for i, char in enumerate(text[:max_length]):
            vector[i] = ord(char) % 100 / 100  # 归一化到0-1之间
    return vector

def create_user_chunks(users: np.ndarray, chunk_size: int) -> List[np.ndarray]:
    """将用户分成多个块"""
    return [users[i:i+chunk_size] for i in range(0, len(users), chunk_size)]

def clean_dataframe(df: pd.DataFrame, event_type: str) -> pd.DataFrame:
    """清理数据DataFrame"""
    # 检查是否有必需的列
    required_cols = ['client_id', 'timestamp']
    if not all(col in df.columns for col in required_cols):
        logger.warning(f"{event_type} 数据缺少必需的列: {required_cols}")
        # 添加缺失的列并填充默认值
        for col in required_cols:
            if col not in df.columns:
                if col == 'client_id':
                    df[col] = -1
                elif col == 'timestamp':
                    df[col] = pd.Timestamp.now()
    
    # 处理时间戳
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    # 替换缺失的时间戳
    if df['timestamp'].isna().any():
        df.loc[df['timestamp'].isna(), 'timestamp'] = pd.Timestamp.now()
    
    # 添加事件类型
    df['event_type'] = event_type
    
    # 处理商品SKU
    if event_type in ['product_buy', 'add_to_cart', 'remove_from_cart']:
        if 'sku' not in df.columns:
            df['sku'] = -1
        # 替换SKU中的无效值
        df['sku'] = df['sku'].fillna(-1)
    elif event_type in ['page_visit', 'search_query']:
        df['sku'] = -1
    
    # 处理URL
    if event_type == 'page_visit':
        if 'url' not in df.columns:
            df['url'] = -1
        # 替换URL中的无效值
        df['url'] = df['url'].fillna(-1)
        # 确保URL为整数类型
        df['url'] = pd.to_numeric(df['url'], errors='coerce').fillna(-1).astype(int)
    
    # 处理搜索查询
    if event_type == 'search_query':
        if 'query' not in df.columns:
            df['query'] = ''
        df['query'] = df['query'].fillna('')
    
    return df

def load_events_parallel(data_dir: Path, current_chunk_users: set) -> Dict[str, pd.DataFrame]:
    """并行加载事件数据"""
    events = {}
    event_types = ['product_buy', 'add_to_cart', 'remove_from_cart', 
                  'page_visit', 'search_query']
    
    def load_event(event_type):
        file_path = data_dir / f"{event_type}.parquet"
        logger.info(f"正在加载 {event_type} 数据，文件路径: {file_path}")
        if not file_path.exists():
            logger.error(f"文件不存在: {file_path}")
            raise FileNotFoundError(f"找不到文件: {file_path}")
        
        try:
            # 1. 首先获取文件元数据
            parquet_file = pq.ParquetFile(file_path)
            columns = parquet_file.schema.names
            
            # 2. 确定需要的列
            needed_columns = ['client_id', 'timestamp']
            if event_type in ['product_buy', 'add_to_cart', 'remove_from_cart']:
                needed_columns.append('sku')
            elif event_type == 'search_query':
                needed_columns.append('query')
            elif event_type == 'page_visit':
                needed_columns.append('url')  # 确保加载URL字段
            
            # 过滤出文件中存在的列
            columns_to_read = [col for col in needed_columns if col in columns]
            
            try:
                # 尝试直接用PyArrow过滤
                batches = []
                for batch in parquet_file.iter_batches(
                    batch_size=100000,
                    columns=columns_to_read,
                    filters=[('client_id', 'in', list(current_chunk_users))]
                ):
                    batches.append(batch)
                
                if batches:
                    table = pa.Table.from_batches(batches)
                    df = table.to_pandas()
                else:
                    df = pd.DataFrame(columns=columns_to_read)
            except Exception:
                # 回退到读取全部后在内存中过滤
                batches = []
                for batch in parquet_file.iter_batches(batch_size=100000, columns=columns_to_read):
                    batch_df = batch.to_pandas()
                    batch_df = batch_df[batch_df['client_id'].isin(current_chunk_users)]
                    if not batch_df.empty:
                        batches.append(pa.Table.from_pandas(batch_df))
                
                if batches:
                    table = pa.concat_tables(batches)
                    df = table.to_pandas()
                else:
                    df = pd.DataFrame(columns=columns_to_read)
            
            # 清理数据
            df = clean_dataframe(df, event_type)
            
            logger.info(f"{event_type} 数据加载完成，形状: {df.shape}")
            return event_type, df
        except Exception as e:
            logger.error(f"加载 {event_type} 数据时发生错误: {str(e)}")
            return event_type, pd.DataFrame(columns=['client_id', 'timestamp'])
    
    # 并行加载所有事件数据
    with ThreadPoolExecutor(max_workers=len(event_types)) as executor:
        future_to_event = {executor.submit(load_event, event_type): event_type 
                         for event_type in event_types}
        
        for future in as_completed(future_to_event):
            event_type, df = future.result()
            events[event_type] = df
    
    return events

def load_product_properties(data_dir: Path) -> pd.DataFrame:
    """加载商品属性数据"""
    file_path = data_dir / "product_properties.parquet"
    logger.info(f"正在加载商品属性数据，文件路径: {file_path}")
    
    try:
        if not file_path.exists():
            logger.error(f"文件不存在: {file_path}")
            raise FileNotFoundError(f"找不到文件: {file_path}")
        
        # 使用pyarrow直接读取
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        # 清理数据
        df = df.fillna({'name': '', 'category': 0, 'price': 0})
        
        logger.info(f"商品属性数据加载完成，形状: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"加载商品属性数据时发生错误: {str(e)}")
        # 创建一个默认的空DataFrame
        return pd.DataFrame({'sku': [], 'name': [], 'category': [], 'price': []}) 
    



