import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch import Tensor

class LightGCN(MessagePassing):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int, num_layers: int):
        super().__init__(aggr='add') # 聚合方式：求和（LightGCN默认）
        # 初始化 User 和 Item 的“身份证”（Embedding层）
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # 初始化 Embedding (随机初始化)
        self.users_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.items_emb = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)
        
        # 使用正态分布初始化权重 (参考原论文)
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

    def forward(self, edge_index: Tensor):
        """
        前向传播：
        edge_index: 图的边索引，形状为 [2, num_edges]
        返回：用户和物品的最终嵌入表示
        """
        # 1. 🆔 你是谁？：拼接用户和物品的嵌入，准备出发
        x = torch.cat([self.users_emb.weight, self.items_emb.weight], dim=0)
        
        # 2. ⚖️ 归一化：防止热门节点（比如热门电影）的特征被过度放大
        edge_index_norm, edge_weight = gcn_norm(edge_index, num_nodes=x.size(0), add_self_loops=False)
        embs = [x]  # 保存每一层的嵌入结果（用于后续层融合）
        
        # 3. 👥 溜达几圈：多层消息传播
        for _ in range(self.num_layers):
            # propagate：PyG的魔法方法，自动处理消息传递（不用自己写循环）
            x = self.propagate(edge_index_norm, x=x, edge_weight=edge_weight)
            embs.append(x)
            
        # 4. 📊 层融合：把每一层的结果平均，得到最终嵌入
        embs = torch.stack(embs, dim=1)
        final_emb = torch.mean(embs, dim=1)
        
        # 拆分回 User 和 Item
        users, items = torch.split(final_emb, [self.num_users, self.num_items])
        return users, items

    # 消息传递逻辑（PyG固定写法，新手不用深究，复制即可）
    def message(self, x_j, edge_weight):
        # x_j：邻居节点的特征，edge_weight：归一化权重
        return edge_weight.view(-1, 1) * x_j  
    