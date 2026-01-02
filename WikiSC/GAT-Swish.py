# gnn_compare_stability_speed_wikics_coauthor.py
# Baseline-GAT vs MixGAT on WikiCS / Coauthor-CS / Coauthor-Physics
# 说明：
# - 单图、半监督单标签分类（CrossEntropy + Accuracy）
# - WikiCS：20 套固定划分；使用 --split_id 选择
# - Coauthor-*：原始不含划分；提供 perclass / ratio 两种划分方式
# - 继续提供 AULC / E_abs / T_abs / E_tau / T_tau 收敛速度指标（基于 val_acc）

import os, time, math, argparse, random, warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.transforms import NormalizeFeatures, ToUndirected
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.datasets import WikiCS, Coauthor

# -----------------------------
# 工具：种子、设备与计时
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def now_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# -----------------------------
# 指标：Dirichlet 能量（过平滑度量）
# -----------------------------
@torch.no_grad()
def dirichlet_energy(X: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> float:
    row, col = edge_index
    if edge_weight is None:
        w = torch.ones(row.size(0), device=X.device, dtype=X.dtype)
    else:
        w = edge_weight
    diff = X[row] - X[col]
    e = 0.5 * (w[:, None] * (diff ** 2)).sum()
    return float(e.item())


# -----------------------------
# 梯度向量拼接与余弦
# -----------------------------
def concat_grads(model: nn.Module) -> torch.Tensor:
    vecs = []
    for p in model.parameters():
        if p.grad is not None:
            vecs.append(p.grad.detach().view(-1))
    if not vecs:
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            dev = torch.device('cpu')
        return torch.zeros(1, device=dev)
    return torch.cat(vecs, dim=0)


def cos_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    if a is None or b is None:
        return 0.0
    na = torch.linalg.norm(a)
    nb = torch.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(torch.dot(a, b) / (na * nb))


# -----------------------------
# 模型定义：GAT & MixGAT
# -----------------------------

# -----------------------------
# Swish 激活函数定义
# -----------------------------
class Swish(nn.Module):
    """
    Swish 激活函数：
        f(x) = x * sigmoid(βx)
    当 learnable=True 时，β 为可学习参数（类似 SiLU+β）
    """
    def __init__(self, learnable: bool = False):
        super().__init__()
        self.learnable = learnable
        if learnable:
            self.beta = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("beta", torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


# -----------------------------
# 基础 GAT 层（使用 Swish 激活）
# -----------------------------
class GATLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        heads: int = 1,
        concat: bool = True,
        feat_dropout: float = 0.5,
        attn_dropout: float = 0.0,
        activation=None,
        leaky_slope: float = 0.2,
    ):
        super().__init__()
        self.feat_dropout = nn.Dropout(feat_dropout)
        # ✅ 默认激活为 Swish
        self.activation = activation or Swish()
        self.conv = GATConv(
            in_dim,
            out_dim,
            heads=heads,
            concat=concat,
            dropout=attn_dropout,       # dropout 同时作用于特征和注意力
            negative_slope=leaky_slope,
            add_self_loops=True,
            bias=True,
        )
        self._concat = concat
        self._heads = heads
        self._out_dim = out_dim

    @property
    def out_features(self) -> int:
        return self._out_dim * self._heads if self._concat else self._out_dim

    def forward(self, x, edge_index, edge_weight=None):
        x = self.feat_dropout(x)
        h = self.conv(x, edge_index)
        return self.activation(h) if self.activation is not None else h


# -----------------------------
# Mix GAT 层（使用 Swish 激活）
# -----------------------------
class MixGATLayer(nn.Module):
    """
    X_{l+1} = β * Z + (C - β) * σ(Z),  Z = GAT(X_l)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        heads: int = 1,
        concat: bool = True,
        beta: float = 0.5,
        C: float = 1.2,
        feat_dropout: float = 0.5,
        attn_dropout: float = 0.0,
        activation=None,
        leaky_slope: float = 0.2,
    ):
        super().__init__()
        self.beta = float(beta)
        self.C = float(C)
        self.feat_dropout = nn.Dropout(feat_dropout)
        # ✅ Swish 激活
        self.activation = activation or Swish()
        self.conv = GATConv(
            in_dim,
            out_dim,
            heads=heads,
            concat=concat,
            dropout=attn_dropout,
            negative_slope=leaky_slope,
            add_self_loops=True,
            bias=True,
        )
        self._concat = concat
        self._heads = heads
        self._out_dim = out_dim

    @property
    def out_features(self) -> int:
        return self._out_dim * self._heads if self._concat else self._out_dim

    def forward(self, x, edge_index, edge_weight=None):
        x = self.feat_dropout(x)
        z = self.conv(x, edge_index)
        if self.activation is None:
            return self.C * z
        return self.beta * z + (self.C - self.beta) * self.activation(z)


# -----------------------------
# Baseline GAT 主干（Swish）
# -----------------------------
class BaselineGAT(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        *,
        layers: int = 2,
        dropout: float = 0.5,
        heads_hidden: int = 4,
        attn_dropout: float = 0.0,
        leaky_slope: float = 0.2,
        activation=None,
    ):
        super().__init__()
        assert layers >= 2
        mods = []
        cur_in = in_dim
        for l in range(layers):
            is_last = (l == layers - 1)
            if is_last:
                layer = GATLayer(
                    cur_in, out_dim, heads=1, concat=False,
                    feat_dropout=dropout, attn_dropout=attn_dropout,
                    activation=None, leaky_slope=leaky_slope
                )
            else:
                layer = GATLayer(
                    cur_in, hidden, heads=heads_hidden, concat=True,
                    feat_dropout=dropout, attn_dropout=attn_dropout,
                    activation=activation or Swish(), leaky_slope=leaky_slope
                )
            mods.append(layer)
            cur_in = layer.out_features
        self.layers = nn.ModuleList(mods)

    def forward(self, x, edge_index, edge_weight=None, return_hidden=False):
        hidden_feats = None
        for l, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            if l == len(self.layers) - 2:
                hidden_feats = x
        return (x, hidden_feats) if return_hidden else x


# -----------------------------
# Mix GAT 主干（Swish）
# -----------------------------
class MixGAT(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        *,
        layers: int = 2,
        dropout: float = 0.5,
        beta: float = 0.5,
        C: float = 1.0,
        beta_list: Optional[List[float]] = None,
        heads_hidden: int = 4,
        attn_dropout: float = 0.0,
        leaky_slope: float = 0.2,
        activation=None,
    ):
        super().__init__()
        assert layers >= 2
        if beta_list is not None:
            assert len(beta_list) == layers, f"beta_list 长度({len(beta_list)})必须等于 layers({layers})"
        mods = []
        cur_in = in_dim
        for l in range(layers):
            is_last = (l == layers - 1)
            beta_l = (beta_list[l] if beta_list is not None else beta)
            if is_last:
                layer = MixGATLayer(
                    cur_in, out_dim, heads=1, concat=False,
                    beta=beta_l, C=C,
                    feat_dropout=dropout, attn_dropout=attn_dropout,
                    activation=None, leaky_slope=leaky_slope
                )
            else:
                layer = MixGATLayer(
                    cur_in, hidden, heads=heads_hidden, concat=True,
                    beta=beta_l, C=C,
                    feat_dropout=dropout, attn_dropout=attn_dropout,
                    activation=activation or Swish(), leaky_slope=leaky_slope
                )
            mods.append(layer)
            cur_in = layer.out_features
        self.layers = nn.ModuleList(mods)

    def forward(self, x, edge_index, edge_weight=None, return_hidden=False):
        hidden_feats = None
        for l, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            if l == len(self.layers) - 2:
                hidden_feats = x
        return (x, hidden_feats) if return_hidden else x
        
# -----------------------------
# 数据集加载：WikiCS / Coauthor-CS / Coauthor-Physics
# -----------------------------
def load_wikics(root: str, split_id: int = 0, undirected: bool = True) -> Tuple[Data, torch.Tensor, Optional[torch.Tensor]]:
    ds = WikiCS(root=os.path.join(root, 'wikics'), transform=NormalizeFeatures())
    data = ds[0]
    if undirected:
        data = ToUndirected()(data)
    data.y = data.y.view(-1).long()
    assert data.train_mask.dim() == 2 and data.val_mask.dim() == 2, "WikiCS 的 train/val 掩码应为二维 [N, 20]"
    assert 0 <= split_id < data.train_mask.size(1), f"split_id 越界：{split_id} not in [0, {data.train_mask.size(1)-1}]"
    data.train_mask = data.train_mask[:, split_id]
    data.val_mask   = data.val_mask[:, split_id]
    return data, data.edge_index, None


def _make_masks_perclass(y: torch.Tensor, num_classes: int, train_per_class: int, val_per_class: int, seed: int):
    g = torch.Generator(); g.manual_seed(seed)
    N = y.size(0)
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask   = torch.zeros(N, dtype=torch.bool)
    test_mask  = torch.zeros(N, dtype=torch.bool)
    for c in range(num_classes):
        idx = torch.where(y == c)[0]
        if idx.numel() == 0: continue
        perm = idx[torch.randperm(idx.numel(), generator=g)]
        n_tr = min(train_per_class, perm.numel())
        n_va = min(val_per_class, max(perm.numel() - n_tr, 0))
        tr = perm[:n_tr]; va = perm[n_tr:n_tr+n_va]; te = perm[n_tr+n_va:]
        train_mask[tr] = True; val_mask[va] = True; test_mask[te] = True
    if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
        raise RuntimeError("perclass 划分产生了空的 train/val/test，请调大每类样本数或检查数据。")
    return train_mask, val_mask, test_mask


def _make_masks_ratio(y: torch.Tensor, train_ratio: float, val_ratio: float, seed: int):
    g = torch.Generator(); g.manual_seed(seed)
    N = y.size(0)
    perm = torch.randperm(N, generator=g)
    n_tr = int(N * train_ratio); n_va = int(N * val_ratio)
    tr = perm[:n_tr]; va = perm[n_tr:n_tr+n_va]; te = perm[n_tr+n_va:]
    train_mask = torch.zeros(N, dtype=torch.bool); train_mask[tr] = True
    val_mask   = torch.zeros(N, dtype=torch.bool); val_mask[va]  = True
    test_mask  = torch.zeros(N, dtype=torch.bool); test_mask[te] = True
    if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
        raise RuntimeError("ratio 划分产生了空的 train/val/test，请调整比例。")
    return train_mask, val_mask, test_mask


def load_coauthor(root: str, which: str = 'CS', split_mode: str = 'perclass',
                  train_per_class: int = 20, val_per_class: int = 30,
                  train_ratio: float = 0.6, val_ratio: float = 0.2,
                  seed: int = 0, undirected: bool = True) -> Tuple[Data, torch.Tensor, Optional[torch.Tensor]]:
    assert which in ('CS', 'Physics')
    ds = Coauthor(root=os.path.join(root, f'coauthor_{which.lower()}'), name=which, transform=NormalizeFeatures())
    data = ds[0]
    if undirected:
        data = ToUndirected()(data)
    data.y = data.y.view(-1).long()
    num_classes = int(data.y.max()) + 1

    if split_mode == 'perclass':
        tr_mask, va_mask, te_mask = _make_masks_perclass(data.y, num_classes, train_per_class, val_per_class, seed)
    elif split_mode == 'ratio':
        tr_mask, va_mask, te_mask = _make_masks_ratio(data.y, train_ratio, val_ratio, seed)
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    data.train_mask = tr_mask
    data.val_mask   = va_mask
    data.test_mask  = te_mask
    return data, data.edge_index, None


# -----------------------------
# 评估（自动兼容 1D 或 2D 掩码）
# -----------------------------
@torch.no_grad()
def evaluate_single_graph(model, data, edge_index, edge_weight, split='val'):
    model.eval()
    out = model(data.x, edge_index, edge_weight)
    pred = out.argmax(dim=-1)

    def _pick_mask(mask):
        if mask.dim() == 2:
            return mask[:, 0]
        return mask

    if split == 'val':
        mask = _pick_mask(data.val_mask)
    elif split == 'test':
        mask = _pick_mask(data.test_mask)
    else:
        mask = _pick_mask(data.train_mask)

    correct = (pred[mask] == data.y[mask]).sum().item()
    total = int(mask.sum())
    return correct / max(total, 1)


def nll_loss_for_pyg(out, y, mask):
    return F.cross_entropy(out[mask], y[mask])


@dataclass
class RunConfig:
    model_type: str  # 'baseline' or 'mix'
    seed: int
    dataset: str  # 'WikiCS' | 'CoauthorCS' | 'CoauthorPhysics'
    layers: int = 3
    hidden: int = 256
    dropout: float = 0.4
    lr: float = 0.005
    weight_decay: float = 1e-4
    max_epochs: int = 500
    patience: int = 100
    warmup_epochs: int = 10
    beta: float = 0.9
    C: float = 1.0
    tau: float = 0.90
    abs_target: float = 0.70
    device: str = 'auto'
    beta_list: Optional[List[float]] = None
    beta_mode: str = 'linear'  # 'custom' | 'linear' | 'exp' | 'const'

    # 划分相关
    split_id: int = 0
    split_mode: str = 'perclass'  # 'perclass' | 'ratio'
    train_per_class: int = 20
    val_per_class: int = 30
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    undirected: bool = True

    # GAT 相关
    heads_hidden: int = 4
    attn_dropout: float = 0.0
    leaky_slope: float = 0.2


# 生成 beta 序列
def _build_beta_list(layers: int, beta: float, mode: str) -> List[float]:
    if mode == 'custom':
        return []
    if layers <= 1:
        return [beta]
    if mode == 'const':
        return [beta] * layers
    if mode == 'linear':
        hi, lo = 0.95, beta
        return [float(hi - (hi - lo) * l / (layers - 1)) for l in range(layers)]
    if mode == 'exp':
        hi = 0.95
        gamma = (beta / hi) ** (1 / (layers - 1))
        return [float(hi * (gamma ** l)) for l in range(layers)]
    raise ValueError(f"Unknown beta_mode: {mode}")


def build_model(cfg: RunConfig, in_dim, out_dim, use_cached_ignored: bool):
    # cached 对 GAT 非必需，这里忽略
    betas = None
    auto_betas = _build_beta_list(cfg.layers, cfg.beta, cfg.beta_mode)
    if auto_betas:
        betas = auto_betas
    if cfg.beta_list is not None and len(cfg.beta_list) > 0:
        betas = cfg.beta_list
    if cfg.model_type == 'baseline':
        return BaselineGAT(
            in_dim, cfg.hidden, out_dim,
            layers=cfg.layers, dropout=cfg.dropout,
            heads_hidden=cfg.heads_hidden,
            attn_dropout=cfg.attn_dropout, leaky_slope=cfg.leaky_slope
        )
    elif cfg.model_type == 'mix':
        return MixGAT(
            in_dim, cfg.hidden, out_dim,
            layers=cfg.layers, dropout=cfg.dropout,
            beta=cfg.beta, C=cfg.C, beta_list=betas,
            heads_hidden=cfg.heads_hidden,
            attn_dropout=cfg.attn_dropout, leaky_slope=cfg.leaky_slope
        )
    else:
        raise ValueError('Unknown model_type')


# -----------------------------
# 单图训练主循环（与原版一致）
# -----------------------------
def train_one_run_single_graph(cfg: RunConfig, data: Data, edge_index, edge_weight) -> Tuple[Dict, pd.DataFrame]:
    device = get_device() if cfg.device == 'auto' else torch.device(cfg.device)
    set_seed(cfg.seed)
    num_classes = int(data.y.max().item()) + 1

    model = build_model(cfg, data.num_features, num_classes, use_cached_ignored=True).to(device)
    data = data.to(device)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device) if edge_weight is not None else None

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = -1.0
    best_test_at_val = -1.0
    best_epoch = 0
    no_improve = 0

    rows = []
    timer_started = False
    wall_start = 0.0
    prev_grad_vec = None
    prev_loss = None

    if cfg.warmup_epochs >= cfg.max_epochs:
        warnings.warn("warmup_epochs >= max_epochs：AULC_time 将为 0；请调小 warmup。")

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out, hid = model(data.x, edge_index, edge_weight, return_hidden=True)
        loss = nll_loss_for_pyg(out, data.y, data.train_mask)
        loss.backward()

        grad_vec = concat_grads(model)
        grad_norm = float(torch.linalg.norm(grad_vec).item())
        cos = cos_sim(grad_vec, prev_grad_vec) if prev_grad_vec is not None else 0.0
        prev_grad_vec = grad_vec.detach().clone()

        optimizer.step()

        if epoch == cfg.warmup_epochs + 1 and not timer_started:
            now_synchronize(); wall_start = time.time(); timer_started = True

        val_acc = evaluate_single_graph(model, data, edge_index, edge_weight, split='val')
        test_acc = evaluate_single_graph(model, data, edge_index, edge_weight, split='test')
        train_acc = evaluate_single_graph(model, data, edge_index, edge_weight, split='train')
        d_energy = dirichlet_energy(hid, edge_index, edge_weight) if hid is not None else math.nan

        now_synchronize(); wall = time.time() - wall_start if timer_started else 0.0

        rel_osc = 0.0
        if prev_loss is not None:
            rel_osc = abs(loss.item() - prev_loss) / (prev_loss + 1e-12)
        prev_loss = loss.item()

        rows.append({
            'model_type': cfg.model_type, 'seed': cfg.seed, 'epoch': epoch,
            'wall_time': wall, 'loss': loss.item(),
            'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc,
            'grad_norm': grad_norm, 'grad_cos': cos, 'dirichlet': d_energy
        })

        if val_acc > best_val + 1e-12:
            best_val = val_acc; best_test_at_val = test_acc; best_epoch = epoch; no_improve = 0
        else:
            no_improve += 1
        if no_improve >= cfg.patience:
            break

    df = pd.DataFrame(rows)

    # 收敛速度指标（基于 val_acc）
    aulc_epoch_sum = df['val_acc'].sum()
    aulc_epoch_mean = float(df['val_acc'].mean())
    t = df['wall_time'].to_numpy()
    a = df['val_acc'].to_numpy()
    aulc_time = float(np.trapz(a, t)) if len(df) >= 2 else 0.0

    def first_hit(series, threshold):
        idx = np.where(series >= threshold)[0]
        return int(df.iloc[idx[0]]['epoch']) if len(idx) > 0 else -1

    def first_hit_time(series, threshold):
        idx = np.where(series >= threshold)[0]
        return float(df.iloc[idx[0]]['wall_time']) if len(idx) > 0 else float('inf')

    E_abs = first_hit(df['val_acc'].values, cfg.abs_target)
    T_abs = first_hit_time(df['val_acc'].values, cfg.abs_target)

    a0 = float(df.iloc[0]['val_acc'])
    a_best = float(df['val_acc'].max())
    a_tau = a0 + cfg.tau * (a_best - a0)
    E_tau = first_hit(df['val_acc'].values, a_tau)
    T_tau = first_hit_time(df['val_acc'].values, a_tau)

    grad_var = float(df['grad_norm'].var()) if len(df) > 1 else 0.0
    cos_mean = float(df['grad_cos'].iloc[1:].mean()) if len(df) > 2 else 0.0
    loss_arr = df['loss'].to_numpy()
    osc = float(np.mean(np.abs(loss_arr[1:] - loss_arr[:-1]) / (loss_arr[:-1] + 1e-12))) if len(df) > 2 else 0.0

    summary = {
        **asdict(cfg),
        'best_val': best_val,
        'test_at_best_val': best_test_at_val,
        'best_epoch': best_epoch,
        'E_abs': E_abs, 'T_abs': T_abs,
        'E_tau': E_tau, 'T_tau': T_tau,
        'AULC_epoch_sum': aulc_epoch_sum,
        'AULC_epoch_mean': aulc_epoch_mean,
        'AULC_time': aulc_time,
        'VarGrad': grad_var, 'CosSim_mean': cos_mean, 'Osc_loss': osc,
        'device': str(device),
    }
    return summary, df


# -----------------------------
# 主函数
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CoauthorPhysics',
                        choices=['WikiCS', 'CoauthorCS', 'CoauthorPhysics'])

    # 通用 GNN 超参
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=10)

    # Mix 相关
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--beta_mode', type=str, default='custom', choices=['custom', 'linear', 'exp', 'const'])
    parser.add_argument('--beta_list', type=str, default="0.95,0.9,0.85",
                        help='逗号分隔；留空则按 beta_mode 自动生成')
    parser.add_argument('--C', type=float, default=1)

    # 收敛指标阈值
    parser.add_argument('--tau', type=float, default=0.90)
    parser.add_argument('--abs_target', type=float, default=0.92,
                        help='目标验证精度 (0~1)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2])
    parser.add_argument('--outdir', type=str, default='../runs/data')

    # WikiCS 专用
    parser.add_argument('--split_id', type=int, default=15, help='WikiCS 第几套划分 [0..19]')

    # Coauthor-* 专用划分
    parser.add_argument('--split_mode', type=str, default='perclass', choices=['perclass', 'ratio'])
    parser.add_argument('--train_per_class', type=int, default=20)
    parser.add_argument('--val_per_class', type=int, default=30)
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--val_ratio', type=float, default=0.2)

    # 图方向
    parser.add_argument('--undirected', dest='undirected', action='store_true', help='转无向（默认 True）')
    parser.add_argument('--directed', dest='undirected', action='store_false', help='保持有向（一般不推荐）')
    parser.set_defaults(undirected=True)

    # GAT 特有
    parser.add_argument('--heads_hidden', type=int, default=4, help='隐层注意力头数')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='注意力 dropout（传给 GATConv.dropout）')
    parser.add_argument('--leaky_slope', type=float, default=0.2, help='LeakyRleaky_relu 负斜率（注意力得分）')

    args = parser.parse_args()

    # 解析/生成 beta_list
    beta_list = None
    if args.beta_list is not None and args.beta_list.strip():
        beta_list = [float(x) for x in args.beta_list.split(',')]

    os.makedirs(args.outdir, exist_ok=True)

    all_summaries = []
    all_curves = []

    # 加载数据
    if args.dataset == 'WikiCS':
        data, edge_index, edge_weight = load_wikics(args.outdir, split_id=args.split_id, undirected=args.undirected)
        ds_name = 'WikiCS'
    elif args.dataset == 'CoauthorCS':
        data, edge_index, edge_weight = load_coauthor(
            args.outdir, which='CS', split_mode=args.split_mode,
            train_per_class=args.train_per_class, val_per_class=args.val_per_class,
            train_ratio=args.train_ratio, val_ratio=args.val_ratio,
            seed=args.seeds[0], undirected=args.undirected
        )
        ds_name = 'CoauthorCS'
    else:
        data, edge_index, edge_weight = load_coauthor(
            args.outdir, which='Physics', split_mode=args.split_mode,
            train_per_class=args.train_per_class, val_per_class=args.val_per_class,
            train_ratio=args.train_ratio, val_ratio=args.val_ratio,
            seed=args.seeds[0], undirected=args.undirected
        )
        ds_name = 'CoauthorPhysics'

    print(f"[DEBUG] dataset={ds_name}, undirected={args.undirected}")
    print('N =', data.num_nodes, 'E =', data.num_edges)
    print('x dtype/shape =', data.x.dtype, tuple(data.x.shape))
    print('y dtype/shape =', data.y.dtype, tuple(data.y.shape), 'num_classes =', int(data.y.max())+1)
    print('mask sizes:', int(data.train_mask.sum()), int(data.val_mask.sum()), int(data.test_mask.sum()))
    assert data.y.dtype == torch.long
    assert data.x.dtype in (torch.float32, torch.float64)
    assert not (data.train_mask & data.val_mask).any()
    assert not (data.train_mask & data.test_mask).any()
    assert not (data.val_mask & data.test_mask).any()

    for model_type in ['mix','baseline']:
        for seed in args.seeds:
            cfg = RunConfig(
                model_type=model_type, seed=seed, dataset=ds_name,
                layers=args.layers, hidden=args.hidden, dropout=args.dropout,
                lr=args.lr, weight_decay=args.weight_decay, max_epochs=args.max_epochs,
                patience=args.patience, warmup_epochs=args.warmup_epochs,
                beta=args.beta, beta_list=beta_list if model_type == 'mix' else None,
                beta_mode=args.beta_mode, C=args.C, tau=args.tau, abs_target=args.abs_target,
                split_id=args.split_id, split_mode=args.split_mode,
                train_per_class=args.train_per_class, val_per_class=args.val_per_class,
                train_ratio=args.train_ratio, val_ratio=args.val_ratio,
                undirected=args.undirected,
                heads_hidden=args.heads_hidden,
                attn_dropout=args.attn_dropout,
                leaky_slope=args.leaky_slope,
            )
            summary, df = train_one_run_single_graph(cfg, data, edge_index, edge_weight)
            all_summaries.append(summary); all_curves.append(df)
            print(f"[{ds_name}][{model_type[0:3]}][seed={seed}] best_val={summary['best_val']:.4f} "
                  f"T_abs={summary['T_abs']} E_abs={summary['E_abs']} "
                  f"T_tau={summary['T_tau']:.3f}s E_tau={summary['E_tau']} "
                  f"AULC_time={summary['AULC_time']:.3f} AULC_epoch_mean={summary['AULC_epoch_mean']:.4f}")

    summary_df = pd.DataFrame(all_summaries)
    curves_df = pd.concat(all_curves, axis=0, ignore_index=True)

    summary_csv = os.path.join(args.outdir, f"summary_{ds_name}_GAT_Swish.csv")
    curves_csv = os.path.join(args.outdir, f"curves_{ds_name}_GAT_Swish.csv")
    summary_df.to_csv(summary_csv, index=False)
    curves_df.to_csv(curves_csv, index=False)

    def agg(group):
        t_abs = group['T_abs'].replace(float('inf'), np.nan)
        t_tau = group['T_tau'].replace(float('inf'), np.nan)
        return pd.Series({
            'best_val_mean': group['best_val'].mean(),
            'best_val_std': group['best_val'].std(),
            'T_abs_median': t_abs.median(),
            'T_abs_hit_rate': t_abs.notna().mean(),
            'T_tau_median': t_tau.median(),
            'T_tau_hit_rate': t_tau.notna().mean(),
            'AULC_time_mean': group['AULC_time'].mean(),
            'AULC_time_std': group['AULC_time'].std(),
            'AULC_epoch_mean_mean': group['AULC_epoch_mean'].mean(),
            'VarGrad_mean': group['VarGrad'].mean(),
            'CosSim_mean': group['CosSim_mean'].mean(),
            'Osc_loss_mean': group['Osc_loss'].mean(),
        })

    table = summary_df.groupby('model_type').apply(agg)
    print("=== Aggregate (robust stats over seeds) ===")
    print(table)

    print(f"Saved summary to: {summary_csv}")
    print(f"Saved curves to:  {curves_csv}")


if __name__ == '__main__':
    main()
