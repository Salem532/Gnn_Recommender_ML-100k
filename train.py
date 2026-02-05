import argparse
import torch
import os
from torch.utils.data import DataLoader, TensorDataset
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.utils.common import seed_everything
from src.data.dataset import MovieLensDataset
from src.data.transforms import get_bipartite_edge_index
from src.models.lightgcn import LightGCN
from src.engine.trainer import Trainer
from src.engine.evaluator import Evaluator


def main():
    # 1. 初始化配置
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    
    config = Config(args.config)
    seed_everything(config.data.seed)
    logger = setup_logger("train", config.trainer.log_save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")
    
    # 2. 数据准备
    dataset = MovieLensDataset(config.data.root)
    dataset.download() # 自动处理下载
    dataset.process()  # 自动处理数据
    
    num_users, num_items = dataset.num_users, dataset.num_items
    train_edge_index, test_edge_index = dataset.train_test_split
    
    train_edge_index = train_edge_index.to(device)
    test_edge_index = test_edge_index.to(device)
    
    logger.info(f"Data Loaded: Users={num_users}, Items={num_items}")
    logger.info(f"Train Edges: {train_edge_index.size(1)}, Test Edges: {test_edge_index.size(1)}")
    
    # 3. 构建图结构 (关键步骤)
    # 为 GCN 构建包含双向边的二部图
    graph_edge_index = get_bipartite_edge_index(train_edge_index, num_users, num_items).to(device)
    
    # 4. 初始化模型
    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=config.model.embedding_dim,
        num_layers=config.model.num_layers
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.trainer.learning_rate)
    
    # 5. 初始化训练器与评估器
    trainer = Trainer(model, optimizer, config, device, logger)
    evaluator = Evaluator(k=config.trainer.top_k, logger=logger)
    
    # 准备负采样所需的字典
    train_user_pos = {}
    u_list = train_edge_index[0].cpu().numpy()
    i_list = train_edge_index[1].cpu().numpy()
    for u, i in zip(u_list, i_list):
        if u not in train_user_pos: train_user_pos[u] = set()
        train_user_pos[u].add(i)
        
    # 6. 训练循环
    best_ndcg = 0.0
    # DataLoader 只需要提供训练集的边即可
    train_dataset = TensorDataset(train_edge_index.t())
    train_loader = DataLoader(train_dataset, batch_size=config.trainer.batch_size, shuffle=True)
    
    for epoch in range(1, config.trainer.epochs + 1):
        loss = trainer.train_epoch(train_loader, graph_edge_index, train_user_pos)
        
        if epoch % config.trainer.eval_step == 0:
            metrics = evaluator.evaluate(model, graph_edge_index, train_edge_index, test_edge_index)
            logger.info(f"Epoch {epoch} | Loss: {loss:.4f} | Recall: {metrics[f'Recall@{config.trainer.top_k}']:.4f} | NDCG: {metrics[f'NDCG@{config.trainer.top_k}']:.4f}")
            
            if metrics[f'NDCG@{config.trainer.top_k}'] > best_ndcg:
                best_ndcg = metrics[f'NDCG@{config.trainer.top_k}']
                save_path = os.path.join(config.trainer.model_save_dir, "best_model.pth")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                logger.info("New best model saved.")
                
    logger.info("Training Finished.")

if __name__ == "__main__":
    main()