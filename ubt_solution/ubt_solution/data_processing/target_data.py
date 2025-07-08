import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TargetData:
    def __init__(self, data_dir: Path, target_mode: str = 'all'):
        self.data_dir = data_dir
        self.target_mode = target_mode
        try:
            # 加载训练和验证目标数据
            train_file = data_dir / 'target' / 'train_target.parquet'
            val_file = data_dir / 'target' / 'validation_target.parquet'
            
            if not train_file.exists() or not val_file.exists():
                logger.warning("目标数据文件不存在，将使用空DataFrame")
                self.train_df = pd.DataFrame(columns=["client_id", "category", "sku"])
                self.validation_df = pd.DataFrame(columns=["client_id", "category", "sku"])
            else:
                self.train_df = pd.read_parquet(train_file)
                self.validation_df = pd.read_parquet(val_file)
                self.train_df = pd.concat([self.train_df, self.validation_df], ignore_index=True)
                self.validation_df = self.train_df
            
            # 加载倾向性目标和流行度数据
            propensity_attrs = {
                'propensity_category': (data_dir / 'target' / 'propensity_category.npy', np.int32),
                'propensity_sku': (data_dir / 'target' / 'propensity_sku.npy', np.int32),
                'popularity_category': (data_dir / 'target' / 'popularity_propensity_category.npy', np.float32),
                'popularity_sku': (data_dir / 'target' / 'popularity_propensity_sku.npy', np.float32),
            }
            for attr_name, (path, dtype) in propensity_attrs.items():
                if path.exists():
                    setattr(self, attr_name, np.load(path))
                else:
                    logger.warning(f"文件不存在: {path}")
                    setattr(self, attr_name, np.zeros(0, dtype=dtype))
        
        except Exception as e:
            logger.error(f"加载目标数据时发生错误: {str(e)}")
            # 设置默认空值
            self.train_df = pd.DataFrame(columns=["client_id", "category", "sku"])
            self.validation_df = pd.DataFrame(columns=["client_id", "category", "sku"])
            self.propensity_category = np.array([0], dtype=np.int32)
            self.propensity_sku = np.array([0], dtype=np.int32)
            self.popularity_category = np.array([0], dtype=np.float32)
            self.popularity_sku = np.array([0], dtype=np.float32)
        
    def get_churn_target(self, client_id: int) -> float:
        """计算流失目标值"""
        try:
            if self.target_mode == 'train':
                df = self.train_df
            elif self.target_mode == 'valid':
                df = self.validation_df
            else:
                df = pd.concat([self.train_df, self.validation_df], ignore_index=True)
            return 1.0 if df.loc[df["client_id"] == client_id].empty else 0.0
        except Exception as e:
            logger.error(f"计算流失目标值时发生错误: {str(e)}")
            return 0.0
        
    def get_category_propensity_target(self, client_id: int) -> np.ndarray:
        """计算品类倾向性目标值"""
        try:
            target = np.zeros(len(self.propensity_category), dtype=np.float32)
            # 确保目标不为空
            if len(self.propensity_category) == 0:
                return np.zeros(1, dtype=np.float32)
            
            if self.target_mode == 'train':
                df = self.train_df
            elif self.target_mode == 'valid':
                df = self.validation_df
            else:
                df = pd.concat([self.train_df, self.validation_df], ignore_index=True)
            
            # 获取用户在目标数据中的品类
            cats_in_target = df.loc[df["client_id"] == client_id]["category"].unique()
            
            # 防止索引错误
            if len(cats_in_target) > 0:
                valid_indices = np.isin(self.propensity_category, cats_in_target, assume_unique=True)
                target[valid_indices] = 1
            
            # 确保返回一维数组
            return target.flatten()
        except Exception as e:
            logger.error(f"计算品类倾向性目标值时发生错误: {str(e)}")
            return np.zeros(len(self.propensity_category), dtype=np.float32)
        
    def get_product_propensity_target(self, client_id: int) -> np.ndarray:
        """计算产品倾向性目标值"""
        try:
            target = np.zeros(len(self.propensity_sku), dtype=np.float32)
            # 确保目标不为空
            if len(self.propensity_sku) == 0:
                return np.zeros(1, dtype=np.float32)
            
            if self.target_mode == 'train':
                df = self.train_df
            elif self.target_mode == 'valid':
                df = self.validation_df
            else:
                df = pd.concat([self.train_df, self.validation_df], ignore_index=True)
            
            # 获取用户在目标数据中的产品
            skus_in_target = df.loc[df["client_id"] == client_id]["sku"].unique()
            
            # 防止索引错误
            if len(skus_in_target) > 0:
                valid_indices = np.isin(self.propensity_sku, skus_in_target, assume_unique=True)
                target[valid_indices] = 1
            
            # 确保返回一维数组
            return target.flatten()
        except Exception as e:
            logger.error(f"计算产品倾向性目标值时发生错误: {str(e)}")
            return np.zeros(len(self.propensity_sku), dtype=np.float32)
        
    def get_category_popularity(self) -> np.ndarray:
        """获取品类流行度数据"""
        return self.popularity_category
        
    def get_product_popularity(self) -> np.ndarray:
        """获取产品流行度数据"""
        return self.popularity_sku 