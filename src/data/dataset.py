import os
import ssl
import urllib.request
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from typing import Optional, Callable, List

# 安全设置：允许加载自定义 PyG 对象
from torch_geometric.data.data import DataTensorAttr
torch.serialization.add_safe_globals([DataTensorAttr])

class MovieLensDataset(InMemoryDataset):
    """
    MovieLens 100k 数据集加载器。
    负责下载、清洗、ID重映射，并生成 PyTorch Geometric 数据对象。
    """
    
    # 原始数据下载链接
    url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"

    def __init__(
        self, 
        root: str, 
        transform: Optional[Callable] = None, 
        pre_transform: Optional[Callable] = None
    ):
        """
        Args:
            root (str): 数据存放根目录
        """
        super().__init__(root, transform, pre_transform)

        # 加载处理好的数据 (.pt 文件)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        
        # 加载元数据 (用户数、物品数)，保存了重映射后的用户总数和电影总数。
        meta = torch.load(os.path.join(self.processed_dir, 'metadata.pt'), weights_only=False)
        self._num_users = meta['num_users']
        self._num_items = meta['num_items']

    @property
    def raw_file_names(self) -> List[str]:
        return ['u.data']

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt', 'metadata.pt']

    def download(self) -> None:
        """
        下载数据集。包含 SSL 上下文修复，解决部分网络环境下的证书报错。
        """
        print(f"Downloading {self.url} to {self.raw_dir}...")
        try:
            download_url(self.url, self.raw_dir)
        except Exception:
            # 如果常规下载失败，尝试禁用 SSL 验证
            print("Standard download failed, trying without SSL verification...")
            context = ssl._create_unverified_context() # 忽略 SSL 证书验证
            data = urllib.request.urlopen(self.url, context=context).read() # 重新下载数据
            with open(os.path.join(self.raw_dir, 'u.data'), 'wb') as f:
                f.write(data)
            print("Download successful.")

    def process(self) -> None:
        """
        处理原始数据：
        1. 读取 CSV
        2. 映射 ID 到 0 ~ N-1
        3. 构建 Edge Index
        4. 保存处理后的文件和元数据
        """
        print("Processing raw data...")
        # 读取 u.data (user_id, item_id, rating, timestamp)
        df = pd.read_csv(
            os.path.join(self.raw_dir, 'u.data'), 
            sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'time']
        ) # 最终得到一个 pandas 的 DataFrame 对象

        # 1. 重新映射 ID (确保 ID 是连续的 0, 1, 2...)
        user_ids = df['user_id'].unique()
        item_ids = df['item_id'].unique()

        num_users = len(user_ids)
        num_items = len(item_ids)

        user_map = {uid: i for i, uid in enumerate(user_ids)}
        item_map = {iid: i for i, iid in enumerate(item_ids)}

        # 2. 构建源节点和目标节点列表
        # 注意：这里我们只存储 User -> Item 的单向边，具体图结构在训练时构建
        src = [user_map[uid] for uid in df['user_id']]
        dst = [item_map[iid] for uid, iid in zip(df['user_id'], df['item_id'])]

        edge_index = torch.tensor([src, dst], dtype=torch.long)

        # 3. 创建 Data 对象
        data = Data(edge_index=edge_index)
        data.num_nodes = num_users + num_items # 这一步其实不准确，PyG会自动推断，但显式写出更好

        # 4. 保存数据
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])
        
        # 5. 保存元数据
        metadata = {'num_users': num_users, 'num_items': num_items}
        torch.save(metadata, os.path.join(self.processed_dir, 'metadata.pt'))
        print("Processing complete.")

    @property
    def num_users(self) -> int:
        return self._num_users

    @property
    def num_items(self) -> int:
        return self._num_items

    # --- 方便的数据集划分 ---
    @property
    def train_test_split(self):
        """
        返回训练集边和测试集边 (8:2 划分)
        """
        if not hasattr(self, '_train_edge_index'):
            # 更规范的访问方式，避免通过 .data 访问内部存储
            edge_index = self.get(0).edge_index
            num_edges = edge_index.size(1)
            
            # 随机打乱
            perm = torch.randperm(num_edges)
            train_size = int(0.8 * num_edges)
            
            self._train_edge_index = edge_index[:, perm[:train_size]]
            self._test_edge_index = edge_index[:, perm[train_size:]]
            
        return self._train_edge_index, self._test_edge_index