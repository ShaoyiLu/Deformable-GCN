# Deformable Graph Convolutional Network (Deformable GCN)

## Introduction

This project implements a reproduction of the paper **"Deformable Graph Convolutional Networks" (Park et al., AAAI 2022)** using PyTorch. The main goal is to address the limitations of standard Graph Neural Networks (GNNs) when applied to **heterophilic graphs**, by learning **node positional embeddings** and applying **deformable convolution kernels** on **latent graphs**.

The project includes:
- Graph smoothing for positional embeddings
- Deformable GCN layers with attention-based aggregation
- Custom graph construction using k-nearest neighbors (kNN)
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
   Includes classification loss, separation loss (`L_sep`) and focus loss (`L_focus`) for better interpretability and training stability.

---

## Project Structure

