# gnn_compare_stability_speed.py
# 比较 Baseline-GAT 与 MixGAT 的收敛速度与稳定性（多指标、多随机种子）
import os, time, math, argparse, random, warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.transforms import NormalizeFeatures, ToUndirected, Compose
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid, WikipediaNetwork  # chameleon 在此

# 尝试导入 OGB（环境里没装 ogb 也能跑其它数据集）
try:
    from ogb.nodeproppred import PygNodePropPredDataset
    OGB_AVAILABLE = True
except Exception:
    OGB_AVAILABLE = False

# —— 可选：屏蔽 OGB 更新提示（本地已缓存时更干净）——
os.environ.setdefault("OGB_DISABLE_UPDATE", "1")

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
# 模型定义：Baseline-GAT 与 MixGAT
# -----------------------------
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.5, activation=F.elu,
                 heads: int = 1, concat: bool = True, attn_dropout: float = 0.0,
                 negative_slope: float = 0.2, add_self_loops: bool = True):
        super().__init__()
        self.conv = GATConv(
            in_dim, out_dim,
            heads=heads, concat=concat,
            negative_slope=negative_slope,
            dropout=attn_dropout,
            add_self_loops=add_self_loops,
            bias=True
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x, edge_index, edge_weight_ignored=None):
        x = self.dropout(x)
        h = self.conv(x, edge_index)  # GATConv 不使用 edge_weight
        return self.activation(h) if self.activation is not None else h


class MixGATLayer(nn.Module):
    """
    X_{l+1} = beta * Z + (C - beta) * sigma(Z),  Z = GATConv(X_l)
    """
    def __init__(self, in_dim, out_dim, beta=0.5, C=1.0, dropout=0.5, activation=F.elu,
                 heads: int = 1, concat: bool = True, attn_dropout: float = 0.0,
                 negative_slope: float = 0.2, add_self_loops: bool = True):
        super().__init__()
        self.conv = GATConv(
            in_dim, out_dim,
            heads=heads, concat=concat,
            negative_slope=negative_slope,
            dropout=attn_dropout,
            add_self_loops=add_self_loops,
            bias=True
        )
        self.beta = float(beta)
        self.C = float(C)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x, edge_index, edge_weight_ignored=None):
        x = self.dropout(x)
        z = self.conv(x, edge_index)  # 不使用 edge_weight
        if self.activation is None:
            return self.C * z  # 最后一层：只做缩放，不改变 argmax
        return self.beta * z + (self.C - self.beta) * self.activation(z)


class BaselineGAT(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.5, layers=2, activation=F.elu,
                 heads: int = 1, concat: bool = True, attn_dropout: float = 0.0,
                 negative_slope: float = 0.2, add_self_loops: bool = True):
        super().__init__()
        assert layers >= 2
        mods = []
        dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        for l in range(layers):
            idim, odim = dims[l], dims[l + 1]
            act = activation if l < layers - 1 else None
            mods.append(GATLayer(
                idim, odim, dropout=dropout, activation=act,
                heads=heads if l < layers - 1 else 1,            # 最后一层一般用单头分类（可按需改）
                concat=concat if l < layers - 1 else True,       # 最后一层 logits 维度=out_dim
                attn_dropout=attn_dropout,
                negative_slope=negative_slope,
                add_self_loops=add_self_loops
            ))
        self.layers = nn.ModuleList(mods)

    def forward(self, x, edge_index, edge_weight, return_hidden=False):
        hidden_feats = None
        for l, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            if l == len(self.layers) - 2:  # 倒数第二层输出，作为过平滑监控的特征
                hidden_feats = x
        return (x, hidden_feats) if return_hidden else x


class MixGAT(nn.Module):
    """
    支持每层不同的 beta：
      - 若传入 beta_list（长度==layers），逐层使用；
      - 否则使用标量 beta（各层相同）。
    C 仍为全局标量。
    """
    def __init__(self, in_dim, hidden, out_dim, dropout=0.5, layers=2,
                 beta=0.5, C=1.0, activation=F.elu, beta_list: Optional[List[float]] = None,
                 heads: int = 1, concat: bool = True, attn_dropout: float = 0.0,
                 negative_slope: float = 0.2, add_self_loops: bool = True):
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
            mods.append(MixGATLayer(
                idim, odim, beta=beta_l, C=C, dropout=dropout, activation=act,
                heads=heads if l < layers - 1 else 1,
                concat=concat if l < layers - 1 else True,
                attn_dropout=attn_dropout,
                negative_slope=negative_slope,
                add_self_loops=add_self_loops
            ))
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
    pred = out.argmax(dim=-1)
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
    tau: float = 0.95
    abs_target: float = 0.80
    device: str = 'auto'
    beta_list: Optional[List[float]] = None
    beta_mode: str = 'custom'  # 'custom' | 'linear' | 'exp' | 'const'
    # GAT 相关可选项（默认与原实验等价）
    heads: int = 1
    concat: bool = True
    attn_dropout: float = 0.0
    negative_slope: float = 0.2
    add_self_loops: bool = True


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


def build_model(cfg: RunConfig, in_dim, out_dim):
    betas = None
    auto_betas = _build_beta_list(cfg.layers, cfg.beta, cfg.beta_mode)
    if auto_betas:
        betas = auto_betas
    if cfg.beta_list is not None:
        betas = cfg.beta_list  # 显式传入优先
    if cfg.model_type == 'baseline':
        return BaselineGAT(
            in_dim, cfg.hidden, out_dim,
            dropout=cfg.dropout, layers=cfg.layers,
            activation=F.elu,
            heads=cfg.heads, concat=cfg.concat,
            attn_dropout=cfg.attn_dropout,
            negative_slope=cfg.negative_slope,
            add_self_loops=cfg.add_self_loops
        )
    elif cfg.model_type == 'mix':
        return MixGAT(
            in_dim, cfg.hidden, out_dim,
            dropout=cfg.dropout, layers=cfg.layers,
            beta=cfg.beta, C=cfg.C, beta_list=betas, activation=F.elu,
            heads=cfg.heads, concat=cfg.concat,
            attn_dropout=cfg.attn_dropout,
            negative_slope=cfg.negative_slope,
            add_self_loops=cfg.add_self_loops
        )
    else:
        raise ValueError('Unknown model_type')


# -----------------------------
# 数据加载：统一入口
# -----------------------------
def load_dataset(name: str, outdir: str):
    """
    返回：
      data: PyG Data，带 x, y, train/val/test_mask
      edge_index: 图边（可能已转无向）
      edge_weight: 对 GAT 为 None
    """
    name_low = name.lower()
    edge_weight = None

    if name_low in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=os.path.join(outdir, 'data'), name=name, transform=NormalizeFeatures())
        data = dataset[0]
        edge_index = data.edge_index
        return data, edge_index, edge_weight

    if name_low in ['chameleon', 'cham', 'camel', 'chm']:
        dataset = WikipediaNetwork(
            root=os.path.join(outdir, 'data'),
            name='chameleon',
            transform=Compose([NormalizeFeatures(), ToUndirected()])
        )
        data = dataset[0]
        edge_index = data.edge_index
        return data, edge_index, edge_weight

    if name_low in ['ogbn-arxiv', 'arxiv', 'ogbnarxiv']:
        if not OGB_AVAILABLE:
            raise RuntimeError("选择了 ogbn-arxiv，但未安装 ogb。请先执行：pip install ogb")
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=os.path.join(outdir, 'data'))
        data = dataset[0]
        # ogbn-arxiv 是有向图（引用 -> 被引），常规把边转无向提升传播稳定性
        data.edge_index = ToUndirected()(data).edge_index
        data = NormalizeFeatures()(data)

        # 官方划分（idx -> bool mask）
        split_idx = dataset.get_idx_split()
        n = data.num_nodes
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask   = torch.zeros(n, dtype=torch.bool)
        test_mask  = torch.zeros(n, dtype=torch.bool)
        train_mask[split_idx['train']] = True
        val_mask[split_idx['valid']]   = True
        test_mask[split_idx['test']]   = True

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        edge_index = data.edge_index
        return data, edge_index, edge_weight

    raise ValueError(f"未知数据集：{name}. 支持：Cora/CiteSeer/PubMed/chameleon/ogbn-arxiv")


# -----------------------------
# 单次运行
# -----------------------------
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
        'VarGrad': grad_var, 'CosSim_mean': cos_mean, 'Osc_loss': osc,
        'device': str(device),
    }
    return summary, df


# -----------------------------
# 主入口
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv',
                        choices=['Cora', 'CiteSeer', 'PubMed', 'chameleon', 'ogbn-arxiv'])
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--warmup_epochs', type=int, default=10)

    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--beta_mode', type=str, default='custom', choices=['custom', 'linear', 'exp', 'const'])
    parser.add_argument('--beta_list', type=str, default="0.95,0.89,0.83")
    parser.add_argument('--C', type=float, default=1.0)

    parser.add_argument('--tau', type=float, default=0.95)
    parser.add_argument('--abs_target', type=float, default=0.6)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9])
    parser.add_argument('--outdir', type=str, default='runs')

    # GAT 可选项（默认等价 heads=1, concat=True）
    parser.add_argument('--heads', type=int, default=1)
    parser.add_argument('--concat', action='store_true', default=True, help='中间层是否拼接多头输出')
    parser.add_argument('--no_concat', dest='concat', action='store_false', help='中间层不拼接（会均值/求和）')
    parser.add_argument('--attn_dropout', type=float, default=0.0)
    parser.add_argument('--neg_slope', type=float, default=0.2)
    parser.add_argument('--no_self_loops', action='store_true', help='关闭 GATConv 内部自环')

    args = parser.parse_args()

    # 数据
    data, edge_index, edge_weight = load_dataset(args.dataset, args.outdir)

    # 打印边对称性
    inspect_edge_symmetry(edge_index, num_nodes=data.num_nodes, verbose=True)

    # 解析/生成 beta_list
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

    # 训练两个模型
    all_summaries, all_curves = [], []
    for model_type in ['mix', 'baseline']:
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
                heads=args.heads,
                concat=args.concat,
                attn_dropout=args.attn_dropout,
                negative_slope=args.neg_slope,
                add_self_loops=not args.no_self_loops
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

    summary_csv = os.path.join(args.outdir, f"GAT_summary_{args.dataset}.csv")
    curves_csv = os.path.join(args.outdir, f"curves_{args.dataset}.csv")
    summary_df.to_csv(summary_csv, index=False)
    curves_df.to_csv(curves_csv, index=False)

    # 更稳健的聚合：median + 命中率
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
