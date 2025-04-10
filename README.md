# Deformable Graph Convolutional Network

## Introduction

This project implements a reproduction of the paper **"Deformable Graph Convolutional Networks"** using PyTorch. The main goal is to address the limitations of standard Graph Neural Networks when applied to heterophilic graphs, by learning node positional embeddings and applying deformable convolution kernels on latent graphs.

It includes:
- Graph smoothing for positional embeddings
- Deformable GCN layers with attention-based aggregation
- Custom graph construction using KNN
- Regularization terms to stabilize learning

---

## Working Principle

The key components of this Deformable GCN implementation are:

1. **Node Positional Embedding**  
   Learn continuous latent node coordinates (φ) that capture long range dependencies by applying multi-step feature smoothing.

2. **Latent Graph Construction**  
   Latent neighborhoods are constructed by generating multiple kNN graphs from the smoothed node features.

3. **Deformable Graph Convolution**  
   Applies dynamic attention-based aggregation on neighbor features using relation vectors and kernel deformations.

4. **Attention Aggregation**  
   It is a learned attention mechanism fuses information from various latent graph views, enabling adaptive integration across different graph granularities.

5. **Loss Function**  
   It also adds two regularization losses to improve how clearly the model separates classes and to keep the training process more stable.

---

## Model Architecture

### 1. Graph Smoothing Layer (method `graphSmoothing`)
This module applies T-step feature smoothing on the graph to capture long-range dependencies and generate positional embeddings $𝜙$.

At each step $t$, the feature is updated as:

$$
x^{(t)} = \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} x_u^{(t-1)}
$$

The final positional embedding is obtained by averaging over all $T$ steps:

$$
\phi_v = \frac{1}{T} \sum_{t=1}^{T} x_v^{(t)}
$$

- **Input:** Raw node features $x \in \mathbb{R}^{N \times F}$ 
- **Output:** Smoothed embeddings $\phi \in \mathbb{R}^{N \times T \times F}$
---

### 2. Deformable GCN Layers (method `GCNConvolution`)

These layers perform message passing with dynamic attention over latent relation vectors between nodes.

Given a pair of nodes $i$ and $j$, their relation vector is defined as:

$$
r_{ij} = \phi_i - \phi_j
$$

This relation vector is used to compute an attention coefficient $\alpha_{ij}$, which controls how much information node $i$ receives from node $j$.

The attention score is computed as:

$$
\alpha_{ij} = \frac{ \exp\left( \text{MLP}(r_{ij}) \right) }{ \sum_{k \in \mathcal{N}(i)} \exp\left( \text{MLP}(r_{ik}) \right) }
$$

Then, the message from node $j$ to $i$ is modulated as:

$$
m_{ij} = \alpha_{ij} \cdot W x_j
$$

Finally, each node aggregates messages from its neighbors:

$$
x_i^{\text{new}} = \sum_{j \in \mathcal{N}(i)} m_{ij}
$$

Where:
- $\phi_i \in \mathbb{R}^F$ is the smoothed feature of node $i$.
- $W \in \mathbb{R}^{F \times F'}$ is a learnable weight matrix.
- $\mathcal{N}(i)$ is the neighborhood of node $i$.

---

### 3. Custom Regularization Losses (method `DeformableGCN`)

The final loss combines classification with two regularization terms:

$$
L_{total} = L_{Cross Entropy} + α ⋅ L_{sep} + β ⋅ L_{focus}
$$

**Separation Loss** $\mathcal{L}_{\text{sep}}$: Encourages separation between different classes in feature space:

$$
L_{sep} = ∑_{c1 ≠ c2} cos(μ_c1, μ_c2)
$$

**Focus Loss** $\mathcal{L}_{\text{focus}}$: Reduces variation within the same class attention variance:

$$
L_{focus} = ∑{c} ∑{i ∈ C_c} || x_i^att - μ_c^att ||²
$$

Where:
- $\mu_c$ is the average feature vector of class $c$
- $x_i^{\text{att}}$ is the attention-weighted embedding of node $i$

---

## Project Structure

```
project_name/
├── data/                          # Folder for datasets
│   ├── raw/                       # Raw data files (10 Chameleon .npz splits)
│   └── processed/                 # Processed data files
│       └── chameleons.csv         # Data for features of chameleons
│
├── notebooks/                     # Jupyter notebooks for exploration and prototyping
│   └── model_training.ipynb       # Main method for training model
│
├── results/                       # Results of experiments
│   ├── output.txt                 # Output of training and validation accuracy each epoch
│   └── final_test_accuracy.txt    # Final test accuracy result
│
├── src/                           # Source code for the project
│   ├── data_processing/           # Scripts for data loading and preprocessing
│   │   └── load_data.py           # Loading data
│   ├── evaluation/                # Scripts for evaluation and metrics
│   │   └── evaluate.py            # Training and evaluation
│   ├── models/                    # Model definition and loss functions
│   │   └── deformable_gcn.py      # Core Deformable GCN model with smoothing and regularization
│   └── utils/                     # Helper functions or utilities
│       └── knn_graph.py           # Function to build KNN edge index from features
│
├── tests.py                       # Script to verify data integrity and pipeline readiness
├── README.md                      # Project overview, instructions, and explanation
└── requirements.txt               # List of required Python dependencies
```

## How to Run the Code

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Run the Model

Open and run all cells in

```
notebooks/model_training.ipynb
```

Then training log will be saved to
```
results/output.txt
results/final_test_accuracy
```

---

## Parameters

All core parameters are configured inside `model_training.ipynb`. Everyone can freely tune these. According to multiple tests, if the epoch value is larger, the accuracy will increase.

| Name        | Description                              | Default |
|-------------|------------------------------------------|---------|
| `k`         | Number of neighbors for KNN graph        | 16      |
| `layer`     | Layer dimension in the GCN model         | 256     |
| `alpha`     | Weight for separation loss               | 0.001   |
| `beta`      | Weight for focus loss                    | 0.0001  |
| `epochs`    | Total number of training epoch           | 400     |

---

## Sample Output

```
Epoch 102 | Train Accuracy: 60.16% | Validation Accuracy: 60.63%
Epoch 200 | Train Accuracy: 64.84% | Validation Accuracy: 64.88%
...
Final Test Accuracy: 66.67%
```

---

## Testing

Verify that if that data meets the requirements

```bash
python tests.py
```

Good output：
```
all passed
```

---

## Summary

In this project, I aimed to better understand how Deformable Graph Convolutional Networks can improve performance on heterophilic datasets like chameleon I used. By combining positional embeddings and attention-based aggregation over latent graphs, the model learns to extract meaningful patterns even when connected nodes belong to different classes which is a scenario where traditional GCNs usually fail. The implementation is structured for clarity, and I’ve included tools for training, evaluation, and reproducibility.
