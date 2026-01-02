# 注意:
# - 本版将 Backbone 改为 GraphSAGE (SAGEConv)，其余训练/指标逻辑保持不变。
# - GraphSAGE 的聚合方式可选 'mean'/'sum'/'max'，默认 'mean'。如需切换在 SAGELayer/MixSAGELayer 的 aggr 参数修改。
# - edge_weight 对 SAGEConv 不生效，这里为兼容接口仍传递但在 forward 中忽略。

# gnn_compare_stability_speed.py
# 比较 Baseline-SAGE 与 MixSAGE 的收敛速度与稳定性（多指标，多随机种子）
import os, time, math, argparse, random, warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

# 数据集相关
try:
    from ogb.nodeproppred import PygNodePropPredDataset
    OGB_AVAILABLE = True
except Exception:
    OGB_AVAILABLE = False

from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures, ToUndirected, Compose

from torch_geometric.nn import SAGEConv  # ✅ GraphSAGE

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
# E = 1/2 * sum_{(i,j)∈E} w_ij ||x_i - x_j||^2
# -----------------------------
@torch.no_grad()
def dirichlet_energy(X: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> float:
    row, col = edge_index
    if edge_weight is None:
        w = torch.ones(row.size(0), device=X.device, dtype=X.dtype)
    else:
        w = edge_weight
    diff = X[row] - X[col]  # [m, d]
    e = 0.5 * (w[:, None] * (diff ** 2)).sum()
    return float(e.item())


@torch.no_grad()
def inspect_edge_symmetry(edge_index: torch.Tensor, num_nodes: int, verbose: bool = True) -> float:
    """估计边的对称比例（忽略自环），用于提示 Dirichlet 标度解释。"""
    ei = edge_index.detach().cpu()
    row, col = ei[0], ei[1]
    mask = row != col  # 去掉自环
    row, col = row[mask], col[mask]
    if row.numel() == 0:
        if verbose:
            print("[Info] Graph has only self-loops after masking.")
        return 1.0
    ids = (row.long() * num_nodes + col.long()).numpy()
    rev = (col.long() * num_nodes + row.long()).numpy()
    idset = set(ids.tolist())
    sym = sum(1 for r in rev if r in idset)
    ratio = sym / len(ids)
    if verbose:
        print(f"[Info] Edge symmetry ratio (approx, ignore self-loops): {ratio:.3f} "
              f"(1.0≈双向成对；<1.0≈可能单向存储). "
              f"Dirichlet energy is used for **relative** comparison.")
    return ratio


# -----------------------------
# 梯度拼接、相邻方向余弦
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
# 数据加载：ogbn-arxiv / chameleon
# -----------------------------
@torch.no_grad()
def ensure_1d_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if mask.dim() == 2:
        return mask[:, 0].clone()
    return mask.clone()


def load_dataset(name: str, root: str = 'runs/data'):
    """
    返回 (data, edge_index, edge_weight)
    - data: 含 x, y, train/val/test_mask（均为 1D bool）
    - edge_index: 图边（LongTensor [2, E]）
    - edge_weight: 对 SAGE 恒为 None
    """
    nlow = name.lower()

    if nlow in ['ogbn-arxiv', 'ogbnarxiv', 'arxiv']:
        assert OGB_AVAILABLE, "选择了 ogbn-arxiv，但未安装 ogb。请先：pip install ogb"
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=os.path.join(root, 'ogbn-arxiv'))
        data = dataset[0]
        # 标签 [N,1] -> [N]
        if data.y.dim() > 1:
            data.y = data.y.view(-1)

        # 官方划分 -> bool mask
        split = dataset.get_idx_split()
        N = data.num_nodes
        train_mask = torch.zeros(N, dtype=torch.bool)
        val_mask   = torch.zeros(N, dtype=torch.bool)
        test_mask  = torch.zeros(N, dtype=torch.bool)
        train_mask[split['train']] = True
        val_mask[split['valid']]   = True
        test_mask[split['test']]   = True
        data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask

        # 特征归一化（行归一），并把有向图转无向
        if data.x is not None:
            data.x = F.normalize(data.x, p=1, dim=-1)
        data.edge_index = ToUndirected()(data).edge_index

        edge_index = data.edge_index
        edge_weight = None
        return data, edge_index, edge_weight

    if nlow in ['chameleon', 'camel', 'cham']:
        dataset = WikipediaNetwork(
            root=os.path.join(root, 'wikipedia'),
            name='chameleon',
            transform=Compose([NormalizeFeatures(), ToUndirected()]),
        )
        data = dataset[0]
        # 统一掩码；若缺失则随机 60/20/20
        data.train_mask = ensure_1d_mask(getattr(data, 'train_mask', None))
        data.val_mask   = ensure_1d_mask(getattr(data, 'val_mask', None))
        data.test_mask  = ensure_1d_mask(getattr(data, 'test_mask', None))
        if data.train_mask is None or data.val_mask is None or data.test_mask is None:
            N = data.num_nodes
            perm = torch.randperm(N)
            t = int(0.6 * N); v = int(0.8 * N)
            idx_train, idx_val, idx_test = perm[:t], perm[t:v], perm[v:]
            train_mask = torch.zeros(N, dtype=torch.bool); train_mask[idx_train] = True
            val_mask   = torch.zeros(N, dtype=torch.bool); val_mask[idx_val]     = True
            test_mask  = torch.zeros(N, dtype=torch.bool); test_mask[idx_test]   = True
            data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask

        edge_index = data.edge_index
        edge_weight = None
        return data, edge_index, edge_weight

    raise ValueError(f"未知数据集：{name}. 仅支持 ogbn-arxiv / chameleon")


# -----------------------------
# 模型定义：Baseline-SAGE 与 MixSAGE
# -----------------------------
class SAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.5, activation=F.elu, aggr: str = 'mean'):
        super().__init__()
        # SAGEConv: aggr ∈ {'mean','sum','max'}
        self.conv = SAGEConv(in_dim, out_dim, aggr=aggr, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x, edge_index, edge_weight_ignored=None):
        x = self.dropout(x)
        h = self.conv(x, edge_index)  # SAGEConv 不使用 edge_weight
        return self.activation(h) if self.activation is not None else h


class MixSAGELayer(nn.Module):
    """
    X_{l+1} = beta * Z + (C - beta) * sigma(Z),  Z = SAGEConv(X_l)
    """
    def __init__(self, in_dim, out_dim, beta=0.5, C=1.0, dropout=0.5, activation=F.elu, aggr: str = 'mean'):
        super().__init__()
        self.conv = SAGEConv(in_dim, out_dim, aggr=aggr, bias=True)
        self.beta = float(beta)
        self.C = float(C)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x, edge_index, edge_weight_ignored=None):
        x = self.dropout(x)
        z = self.conv(x, edge_index)  # 不使用 edge_weight
        if self.activation is None:
            # 最后一层：C * z（缩放 logits，不改变 argmax）
            return self.C * z
        return self.beta * z + (self.C - self.beta) * self.activation(z)


class BaselineSAGE(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.5, layers=2, activation=F.elu, aggr: str = 'mean'):
        super().__init__()
        assert layers >= 2
        mods = []
        dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        for l in range(layers):
            idim, odim = dims[l], dims[l + 1]
            act = activation if l < layers - 1 else None
            mods.append(SAGELayer(idim, odim, dropout=dropout, activation=act, aggr=aggr))
        self.layers = nn.ModuleList(mods)

    def forward(self, x, edge_index, edge_weight, return_hidden=False):
        hidden_feats = None
        for l, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            if l == len(self.layers) - 2:  # 倒数第二层输出，作为过平滑监控的特征
                hidden_feats = x
        return (x, hidden_feats) if return_hidden else x


class MixSAGE(nn.Module):
    """
    支持每层不同的 beta：
      - 若传入 beta_list（长度==layers），逐层使用；
      - 否则使用标量 beta（各层相同）。
    C 仍为全局标量。
    """
    def __init__(self, in_dim, hidden, out_dim, dropout=0.5, layers=2,
                 beta=0.5, C=1.0, activation=F.elu, beta_list: Optional[List[float]] = None, aggr: str = 'mean'):
        super().__init__()
        assert layers >= 2
        dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]

        if beta_list is not None:
            assert len(beta_list) == layers, f"beta_list 长度({len(beta_list)})必须等于 layers({layers})"
        mods = []
        for l in range(layers):
            idim, odim = dims[l], dims[l + 1]
            act = activation if l < layers - 1 else None
            beta_l = beta_list[l] if beta_list is not None else beta
            mods.append(MixSAGELayer(idim, odim,
                                     beta=beta_l,
                                     C=C,
                                     dropout=dropout,
                                     activation=act,
                                     aggr=aggr))
        self.layers = nn.ModuleList(mods)

    def forward(self, x, edge_index, edge_weight, return_hidden=False):
        hidden_feats = None
        for l, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            if l == len(self.layers) - 2:
                hidden_feats = x
        return (x, hidden_feats) if return_hidden else x


# -----------------------------
# 训练/验证/测试
# -----------------------------
@torch.no_grad()
def evaluate(model, data, edge_index, edge_weight, split='val'):
    model.eval()
    out = model(data.x, edge_index, edge_weight)           # [N, C]
    pred = out.argmax(dim=-1)                              # 多分类
    if split == 'val':
        mask = data.val_mask
    elif split == 'test':
        mask = data.test_mask
    else:
        mask = data.train_mask
    correct = (pred[mask] == data.y[mask]).sum().item()
    total = int(mask.sum())
    return correct / max(total, 1)


def nll_loss_for_pyg(out, y, mask):
    # CrossEntropyLoss（未对数化），out 为 logits
    return F.cross_entropy(out[mask], y[mask])


@dataclass
class RunConfig:
    model_type: str  # 'baseline' or 'mix'
    seed: int
    dataset: str
    layers: int = 2
    hidden: int = 64
    dropout: float = 0.5
    lr: float = 0.01
    weight_decay: float = 5e-4
    max_epochs: int = 500
    patience: int = 50
    warmup_epochs: int = 10
    beta: float = 0.5
    C: float = 1.0
    tau: float = 0.95       # 相对最优命中阈值
    abs_target: float = 0.80  # 绝对阈值命中（可按数据集调整）
    device: str = 'auto'
    beta_list: Optional[List[float]] = None  # 每层 beta（显式传入优先）
    beta_mode: str = 'custom'  # 'custom' | 'linear' | 'exp' | 'const'


def _build_beta_list(layers: int, beta: float, mode: str) -> List[float]:
    """
    自动生成 beta_list 以缓解“layers 与 beta_list 强约束”的易错点。
    - const: 全部等于 beta
    - linear: 从 0.95 均匀下降到 beta
    - exp: 0.95 * gamma^l，gamma 由末层逼近 beta
    - custom: 返回空列表，交由 --beta_list 指定
    """
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


def build_model(cfg: RunConfig, in_dim, out_dim):
    betas = None
    auto_betas = _build_beta_list(cfg.layers, cfg.beta, cfg.beta_mode)
    if auto_betas:
        betas = auto_betas
    if cfg.beta_list is not None:
        betas = cfg.beta_list  # 显式传入优先
    if cfg.model_type == 'baseline':
        return BaselineSAGE(in_dim, cfg.hidden, out_dim, dropout=cfg.dropout, layers=cfg.layers)
    elif cfg.model_type == 'mix':
        return MixSAGE(in_dim, cfg.hidden, out_dim, dropout=cfg.dropout, layers=cfg.layers,
                       beta=cfg.beta, C=cfg.C, beta_list=betas)
    else:
        raise ValueError('Unknown model_type')


def train_one_run(cfg: RunConfig, data, edge_index, edge_weight) -> Tuple[Dict, pd.DataFrame]:
    device = get_device() if cfg.device == 'auto' else torch.device(cfg.device)
    set_seed(cfg.seed)
    model = build_model(cfg, data.num_features, int(data.y.max().item()) + 1).to(device)
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
            now_synchronize()
            wall_start = time.time()
            timer_started = True

        val_acc = evaluate(model, data, edge_index, edge_weight, split='val')
        test_acc = evaluate(model, data, edge_index, edge_weight, split='test')
        train_acc = evaluate(model, data, edge_index, edge_weight, split='train')
        d_energy = dirichlet_energy(hid, edge_index, edge_weight) if hid is not None else math.nan

        now_synchronize()
        wall = time.time() - wall_start if timer_started else 0.0

        # 震荡度
        rel_osc = 0.0
        if prev_loss is not None:
            rel_osc = abs(loss.item() - prev_loss) / (prev_loss + 1e-12)
        prev_loss = loss.item()

        rows.append({
            'model': cfg.model_type, 'seed': cfg.seed, 'epoch': epoch,
            'wall_time': wall, 'loss': loss.item(),
            'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc,
            'grad_norm': grad_norm, 'grad_cos': cos, 'dirichlet': d_energy
        })

        if val_acc > best_val + 1e-12:
            best_val = val_acc
            best_test_at_val = test_acc
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= cfg.patience:
            break

    df = pd.DataFrame(rows)

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
    osc = 0.0
    if len(df) > 2:
        loss_arr = df['loss'].to_numpy()
        osc = float(np.mean(np.abs(loss_arr[1:] - loss_arr[:-1]) / (loss_arr[:-1] + 1e-12)))

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
        'VarGrad': grad_var, 'CosSim_mean': cos_mean,
        'Osc_loss': osc,
        'device': str(device),
    }
    return summary, df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv',
                        choices=['ogbn-arxiv', 'chameleon'])
    parser.add_argument('--layers', type=int, default=7)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--warmup_epochs', type=int, default=10)

    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--beta_mode', type=str, default='custom',
                        choices=['custom', 'linear', 'exp', 'const'],
                        help='自动生成每层 beta（custom 表示使用 --beta_list 或标量 beta）')
    parser.add_argument('--beta_list', type=str, default="0.975,0.9371,0.8733,0.84,0.78,0.7266,0.695")
    parser.add_argument('--C', type=float, default=1.0)

    parser.add_argument('--tau', type=float, default=0.95)
    parser.add_argument('--abs_target', type=float, default=0.6)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2,3])
    parser.add_argument('--outdir', type=str, default='runs')
    args = parser.parse_args()

    # 解析/生成 beta_list：若显式提供则优先；否则按 beta_mode 自动生成
    beta_list = None
    if args.beta_list is not None and args.beta_list.strip():
        beta_list = [float(x) for x in args.beta_list.split(',')]
        if len(beta_list) != args.layers:
            raise ValueError(f"--beta_list 长度({len(beta_list)})必须等于 --layers({args.layers})")
    else:
        auto = _build_beta_list(args.layers, args.beta, args.beta_mode)
        if auto:
            beta_list = auto

    os.makedirs(args.outdir, exist_ok=True)

    # 数据
    data, edge_index, edge_weight = load_dataset(args.dataset, root=os.path.join(args.outdir, 'data'))

    # 一次性打印边对称性（用于 Dirichlet 解释）
    inspect_edge_symmetry(edge_index, num_nodes=data.num_nodes, verbose=True)

    all_summaries = []
    all_curves = []

    # 两个模型：mix & baseline
    for model_type in ['mix','baseline']:
        for seed in args.seeds:
            cfg = RunConfig(
                model_type=model_type,
                seed=seed,
                dataset=args.dataset,
                layers=args.layers,
                hidden=args.hidden,
                dropout=args.dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                max_epochs=args.max_epochs,
                patience=args.patience,
                warmup_epochs=args.warmup_epochs,
                beta=args.beta,
                beta_list=beta_list if model_type == 'mix' else None,
                beta_mode=args.beta_mode,
                C=args.C,
                tau=args.tau,
                abs_target=args.abs_target,
            )
            summary, df = train_one_run(cfg, data, edge_index, edge_weight)
            all_summaries.append(summary)
            all_curves.append(df)

            print(f"[{model_type}][seed={seed}] "
                  f"best_val={summary['best_val']:.4f} "
                  f"T_abs={summary['T_abs']:.3f}s E_abs={summary['E_abs']} "
                  f"T_tau={summary['T_tau']:.3f}s E_tau={summary['E_tau']} "
                  f"AULC_time={summary['AULC_time']:.3f} "
                  f"AULC_epoch_mean={summary['AULC_epoch_mean']:.4f}")

    summary_df = pd.DataFrame(all_summaries)
    curves_df = pd.concat(all_curves, axis=0, ignore_index=True)

    summary_csv = os.path.join(args.outdir, f"SAGE_summary_{args.dataset}.csv")
    curves_csv = os.path.join(args.outdir, f"curves_{args.dataset}.csv")
    summary_df.to_csv(summary_csv, index=False)
    curves_df.to_csv(curves_csv, index=False)

    # 更稳健的聚合：median + 命中率（避免均值掩盖未命中）
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
    print("\n=== Aggregate (robust stats over seeds) ===")
    print(table)

    print(f"\nSaved summary to: {summary_csv}")
    print(f"Saved curves to:  {curves_csv}")


if __name__ == '__main__':
    main()
