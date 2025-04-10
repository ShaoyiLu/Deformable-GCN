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
   Learn continuous latent node coordinates (φ) that capture long range dependencies by applying multi-step feature smoothing.

2. **Latent Graph Construction**  
   Latent neighborhoods are constructed by generating multiple kNN graphs from the smoothed node features.

3. **Deformable Graph Convolution**  
   Applies dynamic attention-based aggregation on neighbor features using relation vectors and kernel deformations.

4. **Attention Aggregation**  
   It is a learned attention mechanism fuses information from various latent graph views, enabling adaptive integration across different graph granularities.

5. **Loss Function**  
   Includes classification loss, separation loss and focus loss to enhance model interpretability and ensure stable training.

---

## Model Architecture

### 1. Graph Smoothing Layer (`graphSmoothing`)
This module applies T-step feature smoothing on the graph to capture long-range dependencies and generate positional embeddings 𝜙.

At each step $\( t \)$, the feature is updated via:

$$
x^{(t)} = \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} x_u^{(t-1)}
$$

The final positional embedding is obtained by averaging over all $\( T \)$ steps:

$$
\phi_v = \frac{1}{T} \sum_{t=1}^{T} x_v^{(t)}
$$

- **Input:** Raw node features $\( x \in \mathbb{R}^{N \times F} \) $ 
- **Output:** Smoothed embeddings $\( \phi \in \mathbb{R}^{N \times T \times F} \)$
---

### 2. Deformable GCN Layers (`GCNConvolution`)

These layers perform message passing with dynamic attention over latent relation vectors between nodes.

Given a pair of nodes $\( i \)$ and $\( j \)$, their relation vector is defined as:

$$
r_{ij} = \phi_i - \phi_j
$$

This relation vector is used to compute an attention coefficient $\( \alpha_{ij} \)$, which controls how much information node $\( i \)$ receives from node $\( j \)$.

The attention score is computed via:

$$
\alpha_{ij} = \text{softmax}_j \left( \text{MLP}(r_{ij}) \right)
$$

Then, the message from node $\( j \)$ to $\( i \)$ is modulated as:

$$
m_{ij} = \alpha_{ij} \cdot W x_j
$$

Finally, each node aggregates messages from its neighbors:

$$
x_i^{\text{new}} = \sum_{j \in \mathcal{N}(i)} m_{ij}
$$

Where:
- $\( \phi_i \in \mathbb{R}^F \)$ is the smoothed feature of node $\( i \)$.
- $\( W \in \mathbb{R}^{F \times F'} \)$ is a learnable weight matrix.
- $\( \mathcal{N}(i) \)$ is the neighborhood of node $\( i \)$.

---

### 3. Custom Regularization Losses (inside `DeformableGCN`)
- **Separation Loss (`l_separation`)**: Encourages class-wise feature vectors to be more separable in feature space using negative cosine similarity between class means
- **Focus Loss (`l_focus`)**: Minimizes intra-class variance in attention-weighted features to make attention more consistent within each class

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

Then training log will be saved to
```
results/output.txt
results/final_test_accuracy
```

---

## Parameters

All core parameters are configured inside `model_training.ipynb`. You can freely tune these.

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

Verify that if the data meets the requirements

```bash
python tests.py
```

Good output：
```
all passed
```

---

## Acknowledgement

This project is developed for exploring graph learning in controlled synthetic environments using Deformable GCN.
