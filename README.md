# Toxic_Comment_Detection
An NLP-based deep learning project to accurately identify toxic comments across 6 categories, using Word2Vec embeddings and a Bi-LSTM architecture. Achieving a high ROC-AUC of 0.97+.

👉 Project Development Pitfalls & Detailed Analysis: [深度学习新手踩坑实录](https://zhuanlan.zhihu.com/p/2001126552239370558)

## 🌟 Key Features

Modular Design: Separated configs, data processing, model, and training logic for high maintainability.

Robust Preprocessing: Custom cleaning pipeline for noisy web comments (handling OOV, tokenization, and padding).

Word2Vec + Bi-LSTM: Combines semantic word vectors with bidirectional context capturing.

Evaluation Toolkit: Includes AUC/F1 reporting and T-SNE visualization of word embeddings.

## 🛠️ 1. Environment Setup

``` bash
# 1. create and activate conda environment
conda create -n comment_det python=3.9 -y
conda activate comment_det

# 2. install dependencies
pip install -r requirements.txt
```

## 📊 2. Dataset Preparation

Download the dataset from Kaggle's [Toxic Comment Classification](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data).

Place the dataset in the following structure (root directory named `data/raw/`)

``` text
data/raw/
├── train.csv          # train set, includes comments and labels
├── test.csv           # test set, includes comments
├── test_labels.csv    # test set labels (for validation only)
└── sample_submission.csv  # sample submission file
```

## 🚀 3. Usage

``` bash
# Train the model with default config.yaml
python train.py

# Visualize Word2Vec embeddings via T-SNE
python visualize.py

# Run full evaluation on the test set
python evaluate.py

# Predict toxicity for a specific comment
python predict.py --text "I need to kill this process."
```


## 🤝 Contributing

Feel free to open issues or pull requests. If this project helped you, please give it a Star ⭐️!

