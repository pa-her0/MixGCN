# MixGCN: Accelerating Graph Convolutional Networks via Adaptive Activation Mixing

This repository contains the official implementation of **MixGCN**, a lightweight and general training framework designed to accelerate the convergence of Graph Convolutional Networks (GCNs) without modifying their architectural backbones.

The code is released to support **reproducibility** of the experiments reported in our paper.

---

## 📄 Paper

**MixGCN: Accelerating Graph Convolutional Networks via Adaptive Activation Mixing**  
*Knowledge-Based Systems (KBS), under review.*

If you use this code in your research, please consider citing:

---

## 🧪 Datasets

Experiments are conducted on standard node classification benchmarks:

* Cora
* CiteSeer
* PubMed
* Coauthor CS
* Coauthor Physics
* ogbn-arxiv

Datasets are automatically downloaded using PyG and OGB utilities.

---


## 📊 Evaluation Metrics

We report both final performance and training efficiency, including:

* Absolute Convergence Time ($T_{\text{abs}}$)
* Epochs to target accuracy
* Area Under Learning Curve (AULC)
* Gradient Variance (VarGrad)
* Gradient Cosine Similarity (CosSim)

These metrics characterize optimization dynamics beyond final accuracy.

---



