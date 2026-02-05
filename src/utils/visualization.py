import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os

def visualize_embeddings(
    user_embeddings: np.ndarray, 
    item_embeddings: np.ndarray, 
    output_path: str,
    method: str = 'tsne'
):
    """
    可视化用户和物品的 Embedding 分布。
    
    Args:
        user_embeddings: 用户嵌入矩阵 [num_samples, dim]
        item_embeddings: 物品嵌入矩阵 [num_samples, dim]
        output_path: 图片保存路径
        method: 降维方法 (目前仅支持 'tsne')
    """
    if method != 'tsne':
        raise NotImplementedError("Currently only t-SNE is supported.")
        
    # 合并数据进行统一降维
    combined = np.vstack([user_embeddings, item_embeddings])
    
    # t-SNE 降维到 2D
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    reduced = tsne.fit_transform(combined)
    
    num_users = user_embeddings.shape[0]
    u_reduced = reduced[:num_users]
    i_reduced = reduced[num_users:]
    
    plt.figure(figsize=(10, 10))
    plt.scatter(u_reduced[:, 0], u_reduced[:, 1], s=20, c='b', alpha=0.5, label='Users')
    plt.scatter(i_reduced[:, 0], i_reduced[:, 1], s=20, c='r', alpha=0.5, label='Items')
    
    plt.legend()
    plt.title("LightGCN Embeddings Visualization (t-SNE)")
    plt.grid(True, alpha=0.3)
    
    # 确保父目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()