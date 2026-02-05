import torch
from torch.optim import Optimizer
from tqdm import tqdm
from src.utils.config import Config
from src.models.lightgcn import LightGCN
from typing import Dict
import numpy as np

class Trainer:
    """
    负责训练循环、BPR 损失计算。
    """
    def __init__(
        self,
        model: LightGCN,
        optimizer: Optimizer,
        config: Config,
        device: torch.device,
        logger
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.logger = logger
        
    def train_epoch(
        self, 
        loader: torch.utils.data.DataLoader, 
        graph_edge_index: torch.Tensor,
        user_pos_dict: Dict[int, set]
    ) -> float:
        """
        训练一个 Epoch。
        
        Args:
            loader: 提供 batch 级的数据 (user, pos_item)
            graph_edge_index: 完整的二部图，用于 GCN 传播
            user_pos_dict: 用于负采样的字典 {user: {pos_items}}
        """
        self.model.train()
        total_loss = 0.0
        
        # 进度条
        pbar = tqdm(loader, desc="Training", leave=False)
        
        for batch in pbar:
            # batch 是 [batch_size, 2]，列 0 是 user，列 1 是 pos_item
            batch = batch[0].to(self.device)
            users = batch[:, 0]
            pos_items = batch[:, 1]
            
            # 1. 负采样 (Negative Sampling)
            neg_items = self._sample_negative(users, user_pos_dict, self.model.num_items)
            
            self.optimizer.zero_grad()
            
            # 2. 模型前向传播 (获取所有节点的 Embedding)
            all_users_emb, all_items_emb = self.model(graph_edge_index)
            
            # 3. 提取当前 Batch 相关的 Embedding
            u_emb = all_users_emb[users]
            pos_emb = all_items_emb[pos_items]
            neg_emb = all_items_emb[neg_items]
            
            # 4. 计算 BPR Loss
            # 正样本得分 = user * pos_item
            pos_scores = torch.mul(u_emb, pos_emb).sum(dim=1)
            # 负样本得分 = user * neg_item
            neg_scores = torch.mul(u_emb, neg_emb).sum(dim=1)
            
            # Softplus 是 log(1 + exp(x))，BPR Loss 是 -log(sigmoid(x))
            # -log(sigmoid(pos - neg)) 等价于 softplus(neg - pos)
            loss = torch.nn.functional.softplus(neg_scores - pos_scores).mean()
            
            # 5. L2 正则化 (只对 Batch 内用到的 Embedding 做正则)
            reg_loss = (1/2) * (u_emb.norm(2).pow(2) + 
                                pos_emb.norm(2).pow(2) + 
                                neg_emb.norm(2).pow(2)) / float(len(users))
            
            loss = loss + self.config.trainer.l2_reg * reg_loss
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        return total_loss / len(loader)
    
    def _sample_negative(self, users, user_pos_dict, num_items):
        """简单且高效的负采样"""
        neg_items = []
        for u in users.cpu().numpy():
            while True:
                # 随机采样一个 item ID
                neg_id = np.random.randint(0, num_items)
                # 确保它不是正样本
                if neg_id not in user_pos_dict[u]:
                    neg_items.append(neg_id)
                    break
        return torch.tensor(neg_items, device=self.device, dtype=torch.long)