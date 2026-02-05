import torch
from torch import Tensor

def get_bipartite_edge_index(edge_index: Tensor, num_users: int, num_items: int) -> Tensor:
    """
    将 [2, E] 的 User-Item 交互边转换为适用于 GCN 的二部图邻接表。
    
    关键逻辑：
    - 用户的节点 ID 范围: [0, num_users - 1]
    - 物品的节点 ID 范围: [num_users, num_users + num_items - 1]
    - 包含双向边: User->Item 和 Item->User
    
    Args:
        edge_index (Tensor): 原始交互边 [2, E]
        num_users (int): 用户总数
        num_items (int): 物品总数
        
    Returns:
        Tensor: 构建好的二部图 edge_index [2, 2*E]
    """
    # 1. 偏移物品 ID
    user_idx = edge_index[0]
    item_idx = edge_index[1] + num_users
    
    # 2. 构建双向边
    # User -> Item
    edge_u_i = torch.stack([user_idx, item_idx], dim=0)
    # Item -> User
    edge_i_u = torch.stack([item_idx, user_idx], dim=0)
    
    # 3. 拼接
    new_edge_index = torch.cat([edge_u_i, edge_i_u], dim=1)
    return new_edge_index