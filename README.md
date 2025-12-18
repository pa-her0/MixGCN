# MixGCN: Accelerating Graph Convolutional Networks via Adaptive Activation Mixing

This repository contains the official implementation of **MixGCN**, a lightweight and general training framework designed to accelerate the convergence of Graph Convolutional Networks (GCNs) without modifying their architectural backbones.

The code is released to support **reproducibility** of the experiments reported in our paper.

---

## 📄 Paper

**MixGCN: Accelerating Graph Convolutional Networks via Adaptive Activation Mixing**  
*Knowledge-Based Systems (KBS), under review.*

If you use this code in your research, please consider citing:

```bibtex
@article{mixgcn2025,
  title   = {MixGCN: Accelerating Graph Convolutional Networks via Adaptive Activation Mixing},
  author  = {Anonymous Authors},
  journal = {Knowledge-Based Systems},
  year    = {2025}
}
````

---

## 📁 Repository Structure

```text

```

---

## ⚙️ Environment Setup

Python ≥ 3.9 is recommended.

```bash
conda create -n mixgcn python=3.10
conda activate mixgcn
pip install -r requirements.txt
```

Main dependencies:

* PyTorch ≥ 2.0
* PyTorch Geometric
* NumPy / SciPy
* scikit-learn

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

## ▶️ Running Experiments

### Example: GCN on Cora with MixGCN

```bash
python scripts/train.py \
  --model gcn \
  --dataset cora \
  --use_mixgcn \
  --activation relu \
  --layers 16
```

### Baseline (without MixGCN)

```bash
python scripts/train.py \
  --model gcn \
  --dataset cora \
  --activation relu \
  --layers 16
```

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

## 🔍 Reproducibility

* Fixed random seeds are used for all experiments
* Hyperparameters follow standard prior work
* Reported results are averaged over multiple runs

Detailed configurations are provided in `experiments/configs/`.

---

## 📌 Notes

* MixGCN focuses on training efficiency rather than architectural novelty
* Compatible with residual connections and normalization layers
* Can be applied to both shallow and deep GNNs

---

## 📬 Contact

Please open an issue for questions or reproducibility concerns.

```


