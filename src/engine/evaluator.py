import torch
import numpy as np
from typing import Dict, List
from src.models.lightgcn import LightGCN

class Evaluator:
    """
    评估器：计算 Recall@K 和 NDCG@K。
    """
    def __init__(self, k: int, logger):
        self.k = k
        self.logger = logger

    @torch.no_grad()
    def evaluate(
        self, 
        model: LightGCN, 
        graph_edge_index: torch.Tensor, 
        train_edge_index: torch.Tensor,
        test_edge_index: torch.Tensor
    ) -> Dict[str, float]:
        """
        执行评估。
        
        Args:
            model: 训练好的模型
            graph_edge_index: 用于模型 Message Passing 的完整图结构
            train_edge_index: 训练集边 (用于 Mask，防止推荐用户已经看过的)
            test_edge_index: 测试集边 (Ground Truth)
        """
        model.eval()
        device = graph_edge_index.device
        
        # 1. 获取所有用户和物品的最终 Embedding
        user_emb, item_emb = model(graph_edge_index)
        
        # 2. 构建 Ground Truth 字典 {user_id: {item_id, ...}}
        test_user_item_set = {}
        test_u = test_edge_index[0].cpu().numpy()
        test_i = test_edge_index[1].cpu().numpy()
        for u, i in zip(test_u, test_i):
            u, i = int(u), int(i) # 显式转为 python int，确保集合查找无误
            if u not in test_user_item_set:
                test_user_item_set[u] = set()
            test_user_item_set[u].add(i)

        # 3. 构建训练集 Mask 字典 (用于过滤)
        train_user_item_set = {}
        train_u = train_edge_index[0].cpu().numpy()
        train_i = train_edge_index[1].cpu().numpy()
        for u, i in zip(train_u, train_i):
            u, i = int(u), int(i) # 显式转为 python int
            if u not in train_user_item_set:
                train_user_item_set[u] = set()
            train_user_item_set[u].add(i)
            
        # 4. 批量计算指标 (防止显存溢出)
        unique_test_users = np.array(list(test_user_item_set.keys()))
        batch_size = 100
        recall_list = []
        ndcg_list = []
        
        for i in range(0, len(unique_test_users), batch_size):
            batch_users = unique_test_users[i : i + batch_size]
            batch_users_tensor = torch.LongTensor(batch_users).to(device)
            
            # 矩阵乘法计算得分 [Batch, Num_Items]
            scores = torch.matmul(user_emb[batch_users_tensor], item_emb.t())
            
            # Mask 掉训练集中出现过的物品 (设为负无穷)
            for idx, user_id in enumerate(batch_users):
                user_id = int(user_id) # 确保类型一致
                if user_id in train_user_item_set:
                    mask_items = list(train_user_item_set[user_id])
                    scores[idx, mask_items] = -float('inf')
            
            # 获取 Top-K
            _, top_k_indices = torch.topk(scores, k=self.k, dim=1)
            top_k_indices = top_k_indices.cpu().numpy()
            
            # 计算指标
            for idx, user_id in enumerate(batch_users):
                ground_truth = test_user_item_set[user_id]
                pred_items = top_k_indices[idx]
                
                # Recall 计算
                hits = sum([1 for item in pred_items if item in ground_truth])
                recall_list.append(hits / len(ground_truth))
                
                # NDCG 计算
                dcg = 0.0
                idcg = 0.0
                for rank, item in enumerate(pred_items):
                    if item in ground_truth:
                        dcg += 1.0 / np.log2(rank + 2)
                for rank in range(min(len(ground_truth), self.k)):
                    idcg += 1.0 / np.log2(rank + 2)
                
                ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)

        return {
            f"Recall@{self.k}": np.mean(recall_list),
            f"NDCG@{self.k}": np.mean(ndcg_list)
        }