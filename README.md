# Gnn_Recommender_ML-100k

A GNN-based recommendation system implemented on the MovieLens-100K dataset, leveraging graph neural network to capture user-item interaction features for personalized item recommendation.

👉 Project Development Pitfalls & Detailed Analysis: [深度学习新手踩坑实录](https://zhuanlan.zhihu.com/p/2002486612634916475)

## 🛠️ 1. Environment Setup

``` bash
# 1. create and activate conda environment
conda create -n comment_det python=3.12 -y
conda activate comment_det

# 2. install dependencies
pip install -r requirements.txt
```

## 📊 2. Dataset Preparation

There is no need to download. This project uses the [MovieLens-100K](https://grouplens.org/datasets/movielens/100k/) dataset. The dataset will be automatically downloaded and preprocessed when you run the training script for the first time.

## 🚀 3. Usage

``` bash
# Train the model with default config.yaml
python train.py

# Visualize results
python visualize.py
```

## 🤝 Contributing

Feel free to open issues or pull requests. If this project helped you, please give it a Star ⭐️!

