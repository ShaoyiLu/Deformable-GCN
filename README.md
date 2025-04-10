# Deformable Graph Convolutional Network

## Introduction

This project implements a reproduction of the paper **"Deformable Graph Convolutional Networks"** using PyTorch. The main goal is to address the limitations of standard Graph Neural Networks when applied to **heterophilic graphs**, by learning node positional embeddings and applying deformable convolution kernels on latent graphs.

This project includes:
- Graph smoothing for positional embeddings
- Deformable GCN layers with attention-based aggregation
- Custom graph construction using KNN
- Regularization terms to stabilize learning

---

## Working Principle

The key components of this Deformable GCN implementation are:

1. **Node Positional Embedding**  
   Uses multi-step feature smoothing to learn continuous latent node coordinates (φ), which capture long-range relationships.

2. **Latent Graph Construction**  
   Based on smoothed features, multiple kNN graphs are generated to define latent neighborhoods for convolution.

3. **Deformable Graph Convolution**  
   Applies dynamic attention-based aggregation on neighbor features using relation vectors and kernel deformations.

4. **Attention Aggregation**  
   Combines multiple latent graph outputs through a learned attention score to adaptively fuse multi-scale information.

5. **Loss Function**  
   Includes classification loss, separation loss and focus loss for better interpretability and training stability.

---

## Project Structure

```
name
├── data/
│   ├── raw/
│   │   └── 10 splits raw .npz files
│   └── processed/
│       └── chameleons.csv
│
├── notebooks/
│   └── model_training.ipynb
│
├── results/
|   ├── final_test_accuracy.txt
│   └── output.txt
│
├── src/
│   ├── data_processing/
│   │   └── load_data.py
│   ├── evaluation/
│   │   └── evaluate.py
│   ├── models/
│   │   └── deformable_gcn.py
│   └── utils/
│       └── knn_graph.py
│
├── tests.py
├── README.md 
└── requirements.txt
```

## How to Run the Code

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Run Model

Open and run all cells in

```
notebooks/model_training.ipynb
```

trianing log will be saved to
```
results/output.txt
results/final_test_accuracy
```

---

## Sample Output

```
Epoch 102 | Train Accuracy: 60.16% | Validation Accuracy: 60.63%
Epoch 200 | Train Accuracy: 64.84% | Validation Accuracy: 64.88%
...
Final Test Accuracy: 66.67%
```

---
