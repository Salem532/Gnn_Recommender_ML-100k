import argparse
import torch
import os
from src.utils.config import Config
from src.models.lightgcn import LightGCN
from src.data.dataset import MovieLensDataset
from src.data.transforms import get_bipartite_edge_index

def run_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", type=int, required=True, help="User ID to recommend for")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = Config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载数据 (需要获取 Item 数量和历史交互以进行过滤)
    dataset = MovieLensDataset(config.data.root)
    # 确保有 train/test 分割，用于构建完整的图
    train_edge, test_edge = dataset.train_test_split
    # 推理时，我们通常利用所有的已知交互来构建图，以获得最准确的 Embedding
    full_edge_index = torch.cat([train_edge, test_edge], dim=1).to(device)
    
    num_users = dataset.num_users
    num_items = dataset.num_items
    
    # 构建图结构
    graph_edge_index = get_bipartite_edge_index(full_edge_index, num_users, num_items).to(device)

    # 2. 加载模型
    model = LightGCN(num_users, num_items, config.model.embedding_dim, config.model.num_layers).to(device)
    model_path = os.path.join(config.trainer.model_save_dir, "best_model.pth")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please train first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. 生成推荐
    with torch.no_grad():
        # 获取所有 Embedding
        user_emb, item_emb = model(graph_edge_index)
        
        target_user = args.user_id
        if target_user >= num_users:
            print(f"Error: User ID {target_user} out of range (Max: {num_users-1})")
            return

        # 计算得分
        u_e = user_emb[target_user] # [dim]
        scores = torch.matmul(u_e, item_emb.t()) # [num_items]
        
        # 过滤已交互物品
        # 找出该用户在原始数据中所有的交互记录
        interacted_mask = full_edge_index[0] == target_user
        interacted_items = full_edge_index[1, interacted_mask]
        
        scores[interacted_items] = -float('inf')
        
        # Top-K
        top_k = 10
        vals, indices = torch.topk(scores, k=top_k)
        
        print(f"--- Recommendations for User {target_user} ---")
        for rank, (score, item_idx) in enumerate(zip(vals, indices)):
            print(f"Rank {rank+1}: Item ID {item_idx.item()} (Score: {score:.4f})")

if __name__ == "__main__":
    run_inference()