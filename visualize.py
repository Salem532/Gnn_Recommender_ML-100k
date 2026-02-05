import argparse
import torch
import os
from src.utils.config import Config
from src.models.lightgcn import LightGCN
from src.data.dataset import MovieLensDataset
from src.utils.visualization import visualize_embeddings
from src.data.transforms import get_bipartite_edge_index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    config = Config(args.config)
    device = torch.device("cpu") # 可视化通常在 CPU 做即可
    
    # 加载数据与模型
    dataset = MovieLensDataset(config.data.root)
    train_edge, _ = dataset.train_test_split
    
    num_users = dataset.num_users
    num_items = dataset.num_items
    
    # 构建图
    graph_edge_index = get_bipartite_edge_index(train_edge, num_users, num_items).to(device)
    
    model = LightGCN(num_users, num_items, config.model.embedding_dim, config.model.num_layers).to(device)
    model_path = os.path.join(config.trainer.model_save_dir, "best_model.pth")
    
    if not os.path.exists(model_path):
        print("Model not found.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 提取 Embedding
    with torch.no_grad():
        u_emb, i_emb = model(graph_edge_index)
        
    # 随机采样一部分点进行可视化 (避免点太多看不清)
    num_samples = 500
    u_indices = torch.randperm(num_users)[:num_samples]
    i_indices = torch.randperm(num_items)[:num_samples]
    
    visualize_embeddings(
        u_emb[u_indices].numpy(),
        i_emb[i_indices].numpy(),
        os.path.join(config.trainer.result_save_dir, "embeddings_tsne.png")
    )
    print("Visualization saved.")

if __name__ == "__main__":
    main()