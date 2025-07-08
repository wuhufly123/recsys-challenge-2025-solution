from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import logging
import numpy as np
from math import log
from tqdm import tqdm
from baseline.aggregated_features_baseline.calculators import (
    Calculator,
    StatsFeaturesCalculator,
    QueryFeaturesCalculator,
)
from baseline.aggregated_features_baseline.constants import (
    EventTypes,
    EVENT_TYPE_TO_COLUMNS,
    QUERY_COLUMN,
    EMBEDDINGS_DTYPE,
)
from datetime import timedelta
from collections import defaultdict
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
window_num=2



# 半衰期（天）
HALF_LIVES = [1, 7, 14, 30, 90, 180]
DECAY_LAMBDAS = [np.log(2) / h for h in HALF_LIVES]  # λ = ln2 / τ
N_EWA = len(HALF_LIVES)        
N_TYPES = 4                      # buy / add / visit / remove

def _exp_decay(days: np.ndarray, lam: float) -> float:
    """单个 λ 的指数衰减和"""
    return float(np.exp(-lam * days).sum())

def get_top_values(events: pd.DataFrame, columns: List[str], top_n: int) -> dict[str, pd.Index]:
    top_values = {}
    for column in columns:
        val_count = events[column].value_counts()
        top_val = val_count.index[:top_n]
        top_values[column] = top_val
    return top_values


class FeaturesAggregator:
    def __init__(self, num_days: List[int], top_n: int, relevant_client_ids: np.ndarray):
        self.num_days = num_days
        self.top_n = top_n
        self._aggregated_features: Dict[int, Dict[EventTypes, np.ndarray]] = {}
        self._features_sizes: Dict[EventTypes, int] = {}
        self._relevant_client_ids = relevant_client_ids
        self._max_timestamp = None  # <-- ADD THIS LINE

    @property
    def _total_dimension(self):
        return sum(self._features_sizes.values())

    def _update_features_sizes(self, event_type: EventTypes, features_size: int):
        self._features_sizes[event_type] = features_size

    def _get_features_size(self, event_type: EventTypes) -> int:
        return self._features_sizes[event_type]

    def _update_features(self, event_type: EventTypes, client_id: int, features: np.ndarray):
        self._aggregated_features.setdefault(client_id, {})[event_type] = features

    def _get_features(self, client_id: int, event_type: EventTypes) -> np.ndarray:
        features_size = self._get_features_size(event_type=event_type)
        return self._aggregated_features.get(client_id, {}).get(
            event_type, np.zeros(features_size, dtype=EMBEDDINGS_DTYPE)
        )

    def get_calculator(self, event_type: EventTypes, df: pd.DataFrame, columns: List[str]) -> Calculator:
        if event_type is EventTypes.SEARCH_QUERY:
            if df.empty:
                dummy_vector = "[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]"
                return QueryFeaturesCalculator(query_column=QUERY_COLUMN, single_query=dummy_vector)
            return QueryFeaturesCalculator(
                query_column=QUERY_COLUMN, single_query=df.iloc[0][QUERY_COLUMN]
            )
        else:
            max_date = df["timestamp"].max()
            unique_values = get_top_values(df, columns, self.top_n)
            return StatsFeaturesCalculator(
                num_days=self.num_days,
                max_date=max_date,
                columns=columns,
                unique_values=unique_values,
            )

    def _filter_events_to_relevant_clients(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["client_id"].isin(self._relevant_client_ids)]

    def generate_features(self, event_type: EventTypes, client_id_column: str, df: pd.DataFrame, columns: List[str]):
        df = self._filter_events_to_relevant_clients(df)

        # Update max timestamp
        if self._max_timestamp is None:
            self._max_timestamp = df["timestamp"].max()
        else:
            self._max_timestamp = max(self._max_timestamp, df["timestamp"].max())

        calculator = self.get_calculator(event_type=event_type, df=df, columns=columns)
        self._update_features_sizes(event_type=event_type, features_size=calculator.features_size)

        for client_id, events in tqdm(df.groupby(client_id_column)):
            assert isinstance(client_id, int)
            features = calculator.compute_features(events=events)
            self._update_features(event_type=event_type, client_id=client_id, features=features)

    def _compute_recency_features(self, all_events: Dict[EventTypes, pd.DataFrame]) -> Dict[int, np.ndarray]:
        recency_features = {cid: np.full(4, 9999, dtype=np.float16) for cid in self._relevant_client_ids}
        
        for idx, etype in enumerate([
            EventTypes.PRODUCT_BUY, 
            EventTypes.ADD_TO_CART, 
            EventTypes.PAGE_VISIT,
            EventTypes.REMOVE_FROM_CART  # ← 新增
        ]):
            df = all_events.get(etype, pd.DataFrame())
            if df.empty:
                continue
            latest_per_client = df.groupby("client_id")["timestamp"].max()

            for client_id, last_time in tqdm(
                list(latest_per_client.items()), 
                desc=f"[Recency] {etype.value}", 
                total=len(latest_per_client)
            ):
                if client_id in recency_features:
                    days_diff = (self._max_timestamp - last_time).days
                    recency_features[client_id][idx] = np.float16(days_diff)

        return recency_features

    def _compute_behavior_counts(self, all_events: Dict[EventTypes, pd.DataFrame]) -> Dict[int, np.ndarray]:
        behavior_counts = {cid: np.zeros(window_num*4, dtype=np.float16) for cid in self._relevant_client_ids}
        for idx, etype in enumerate([
            EventTypes.PRODUCT_BUY, 
            EventTypes.ADD_TO_CART, 
            EventTypes.PAGE_VISIT,
            EventTypes.REMOVE_FROM_CART
        ]):
            df = all_events.get(etype, pd.DataFrame())
            if df.empty:
                continue
            for j, delta in enumerate([7, 30]):
                start_time = self._max_timestamp - pd.Timedelta(days=delta)
                recent_df = df[df["timestamp"] >= start_time]
                grouped = recent_df.groupby("client_id").size()

                for client_id, count in tqdm(
                    list(grouped.items()), 
                    desc=f"[Count {delta}d] {etype.value}", 
                    total=len(grouped)
                ):
                    if client_id in behavior_counts:
                        behavior_counts[client_id][idx * window_num + j] = np.float16(count)
        return behavior_counts

    def _compute_category_entropy(
        self,
        all_events: Dict[EventTypes, pd.DataFrame],
        window: int = 30,
    ) -> Dict[int, np.ndarray]:
        entropy_dict = {cid: np.zeros(1, dtype=np.float16) for cid in self._relevant_client_ids}
        start_time = self._max_timestamp - pd.Timedelta(days=window)

        # 聚合符合时间窗口的事件
        src_events = []
        for etype in [
            EventTypes.PRODUCT_BUY,
            EventTypes.ADD_TO_CART,
            EventTypes.REMOVE_FROM_CART,
        ]:
            df = all_events.get(etype, pd.DataFrame())
            if df.empty:
                continue
            src_events.append(df[df["timestamp"] >= start_time])

        if not src_events:
            return entropy_dict
        last30_df = pd.concat(src_events, ignore_index=True)

        # 不使用 unstack，直接 groupby + defaultdict 汇总
        cat_counts = defaultdict(lambda: defaultdict(int))
        for cid, cat in zip(last30_df["client_id"], last30_df["category"]):
            cat_counts[cid][cat] += 1

        for client_id, cat_count_dict in tqdm(cat_counts.items(), desc="entropy", total=len(cat_counts)):
            if client_id not in entropy_dict:
                continue  # 跳过不在 relevant_id 集合中的 ID
            
            counts = np.array(list(cat_count_dict.values()), dtype=np.float64)
            total = counts.sum()
            if total <= 1:
                continue
            probs = counts / total
            h = -np.sum(probs[probs > 0] * np.log(probs[probs > 0]))
            entropy_dict[client_id][0] = np.float16(np.log1p(h))  # log1p 缩放

        return entropy_dict
    
    def _compute_ewa_counts(self, all_events) -> dict[int, np.ndarray]:
        """
        返回 {client_id: shape [N_TYPES*N_EWA] 的 EWA 计数向量}
        顺序：(buy_τ7, buy_τ30, …, remove_τ180)
        """
        empty_arr = np.zeros(N_TYPES * N_EWA, dtype=np.float16)
        ewa_dict  = {cid: empty_arr.copy() for cid in self._relevant_client_ids}

        etypes = [
            EventTypes.PRODUCT_BUY,
            EventTypes.ADD_TO_CART,
            EventTypes.PAGE_VISIT,
            EventTypes.REMOVE_FROM_CART,
        ]

        for t_idx, etype in enumerate(etypes):
            df = all_events.get(etype, pd.DataFrame())
            if df.empty:
                continue

            df = df[["client_id", "timestamp"]].copy()
            df["days"] = (self._max_timestamp - df["timestamp"]).dt.days.astype(np.float64)

            for lam_idx, lam in enumerate(DECAY_LAMBDAS):
                df_agg = (
                    df.groupby("client_id")["days"]
                    .apply(lambda arr: _exp_decay(arr.to_numpy(), lam))
                )
                offset = t_idx * N_EWA + lam_idx
                desc   = f"[EWA] {etype.value} τ={HALF_LIVES[lam_idx]}d"

                for cid, val in tqdm(df_agg.items(), desc=desc, total=len(df_agg)):
                    if cid in ewa_dict:
                        ewa_dict[cid][offset] = np.float16(val)
        return ewa_dict


    # --------------------------------------------------------
    # 2. 行为比例特征
    # --------------------------------------------------------
    def _compute_behavior_ratios(self, counts_30d) -> dict[int, np.ndarray]:
        """
        输入 counts_30d: {cid: [buy, add, visit, remove]} 30 天内原始计数
        输出 {cid: 5 维比例特征}
        """
        ratio_dict = {}

        for cid, arr in tqdm(counts_30d.items(),
                            desc="[Ratios] 30-day behavior",
                            total=len(counts_30d)):
            buy, add, visit, remove = arr.astype(np.float64)
            ratios = np.array(
                [
                    buy / (visit + 1),
                    add / (visit + 1),
                    buy / (add + 1),
                    remove / (add + 1),
                    arr.sum() / (visit + 1),  # 总活跃度 / 访问
                ],
                dtype=np.float16,
            )
            ratio_dict[cid] = ratios
        return ratio_dict


    # --------------------------------------------------------
    # 3. 生命周期特征
    # --------------------------------------------------------
    def _compute_lifecycle_feats(self, all_events) -> dict[int, np.ndarray]:
        """
        返回 {cid: [user_age, active_span, gap_ratio]}，均 float16
        """
        life_dict = {cid: np.zeros(3, dtype=np.float16)
                    for cid in self._relevant_client_ids}

        # 合并四类事件
        df_list = []
        for etype in [
            EventTypes.PRODUCT_BUY,
            EventTypes.ADD_TO_CART,
            EventTypes.PAGE_VISIT,
            EventTypes.REMOVE_FROM_CART,
        ]:
            df = all_events.get(etype, pd.DataFrame())
            if not df.empty:
                df_list.append(df[["client_id", "timestamp"]])

        if not df_list:
            return life_dict

        all_df  = pd.concat(df_list, ignore_index=True)
        grp_min = all_df.groupby("client_id")["timestamp"].min()
        grp_max = all_df.groupby("client_id")["timestamp"].max()

        for cid in tqdm(grp_min.index,
                        desc="[Lifecycle] user_age / span / gap",
                        total=len(grp_min)):
            first, last = grp_min[cid], grp_max[cid]
            age  = (self._max_timestamp - first).days
            span = (last - first).days
            gap  = (self._max_timestamp - last).days / (age + 1e-6)
            life_dict[cid] = np.array([age, span, gap], dtype=np.float16)

        return life_dict

    def merge_features(self, all_events: Dict[EventTypes, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        recency_dict = self._compute_recency_features(all_events)
        behavior_count_dict = self._compute_behavior_counts(all_events)
        entropy_dict   = self._compute_category_entropy(all_events)
        
        ewa_dict       = self._compute_ewa_counts(all_events)
        
        # 对所有用户的 EWA 特征进行列级 Z-score 标准化
        client_list = list(self._aggregated_features.keys())
        ewa_matrix = np.stack([ewa_dict[cid] for cid in client_list], axis=0).astype(np.float64)
        means = ewa_matrix.mean(axis=0)
        stds = ewa_matrix.std(axis=0)
        stds[stds == 0] = 1
        for cid in client_list:
            ewa_dict[cid] = ((ewa_dict[cid].astype(np.float64) - means) / stds).astype(np.float16)
        
        counts_dict = behavior_count_dict   
        counts30_dict  = {
            cid: counts_dict[cid].reshape(4, -1)[:, 1]  # 取 30d 列
            for cid in counts_dict
        }
        ratio_dict     = self._compute_behavior_ratios(counts30_dict)  # 5 维
        life_dict      = self._compute_lifecycle_feats(all_events)     # 3 维
        
        client_ids = []
        embeddings = []
        for client_id in tqdm(self._aggregated_features.keys()):
            client_ids.append(client_id)
            embeddings_for_client: List[np.ndarray] = []
            for event_type in EVENT_TYPE_TO_COLUMNS.keys():
                features = self._get_features(client_id=client_id, event_type=event_type)
                embeddings_for_client.append(features)

            rec = recency_dict.get(client_id, np.full(4, 9999, dtype=np.float16))
            cnt = behavior_count_dict.get(client_id, np.zeros(window_num*4, dtype=np.float64))
            ent = entropy_dict.get(client_id,  np.zeros(1,  dtype=np.float16)) 
            
            #指数衰减 行为比例 生命周期
            ewa=ewa_dict.get(client_id)     # 16
            ratio=ratio_dict.get(client_id)   # 5
            life=life_dict.get(client_id)    # 3
            
            embeddings_for_client.insert(0, ewa)  # insert ewa at front

            embeddings_for_client.append(np.concatenate([rec, cnt, ent, ratio, life, ewa]))

            embedding_for_client: np.ndarray = np.concatenate(embeddings_for_client)
            embeddings.append(embedding_for_client)

        return np.array(client_ids), np.array(embeddings)

    
    
    