# gnn_compare_stability_speed.py
# 比较 Baseline-ResNet-GNN 与 MixResNet-GNN 的收敛速度与稳定性（多指标、多随机种子）
import os, time, math, argparse, random, warnings
os.environ["OGB_DISABLE_UPDATE_CHECK"] = "1"

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv, SAGEConv, GATConv  # ✅ 可选卷积

# 可选数据集
from torch_geometric.datasets import Planetoid, WikipediaNetwork, HeterophilousGraphDataset
try:
    from ogb.nodeproppred import PygNodePropPredDataset
    OGB_AVAILABLE = True
except Exception:
    OGB_AVAILABLE = False


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
# 掩码/标签 规范化工具
# -----------------------------
def _ensure_1d_bool_mask(mask: torch.Tensor, split_id: int = 0, device=None) -> torch.Tensor:
    """
    将可能为 [N, S] 的掩码规范化为 [N] 的布尔掩码。
    - 若 mask 是二维，则按列 split_id 取一列。
    - 转为 bool，并放到指定 device（如提供）。
    """
    if mask is None:
        return None
    if mask.dim() == 2:
        if not (0 <= split_id < mask.size(1)):
            raise ValueError(f"split_id={split_id} 越界（mask.shape={tuple(mask.shape)}）")
        mask = mask[:, split_id]
    mask = mask.bool()
    if device is not None and mask.device != device:
        mask = mask.to(device)
    return mask


def _ensure_1d_labels(y: torch.Tensor) -> torch.Tensor:
    if y.dim() == 2 and y.size(1) == 1:
        return y.view(-1)
    return y


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


class ConvWrapper(nn.Module):
    """
    统一不同卷积的接口：
      - GCNConv: 支持 edge_weight
      - SAGEConv / GATConv: 忽略 edge_weight
    """
    def __init__(self, conv_type: str, dim: int):
        super().__init__()
        conv_type = conv_type.lower()
        if conv_type == 'gcn':
            self.conv = GCNConv(dim, dim, cached=False, add_self_loops=True, normalize=True)
            self.kind = 'gcn'
        elif conv_type == 'sage':
            self.conv = SAGEConv(dim, dim, normalize=False)
            self.kind = 'sage'
        elif conv_type == 'gat':
            # 设 heads=1，保持输出维度为 dim；可按需改 heads>1 并配合 concat=False
            self.conv = GATConv(dim, dim, heads=1, concat=False)
            self.kind = 'gat'
        else:
            raise ValueError(f"Unknown conv type: {conv_type}")

    def forward(self, x, edge_index, edge_weight=None):
        if self.kind == 'gcn':
            return self.conv(x, edge_index, edge_weight)
        else:
            return self.conv(x, edge_index)



class PostActResBlock(nn.Module):
    """
    后激活残差块（ResNet-GNN 经典版）：
      y = Conv(x, edge_index, edge_weight)
      y = Norm(y)
      y = Act(y)
      out = x + Dropout(y)
    """
    def __init__(self, dim, conv_type='gcn', dropout=0.0, activation=F.relu, norm='layer'):
        super().__init__()
        self.dropout = dropout
        self.activation = activation
        self.norm = nn.LayerNorm(dim) if norm == 'layer' else nn.BatchNorm1d(dim)
        self.conv = ConvWrapper(conv_type, dim)

    def forward(self, x, edge_index, edge_weight=None):
        y = self.conv(x, edge_index, edge_weight)          # ① 卷积
        y = self.norm(y)                                   # ② 归一化（放卷积之后）
        y = self.activation(y) if self.activation is not None else y  # ③ 激活
        y = F.dropout(y, p=self.dropout, training=self.training)
        return x + y                                       # ④ 残差


class PostActResBlockMix(nn.Module):
    """
    后激活残差块（Mix 作用在卷积后的表征上）：
      y = Conv(x)
      z = Norm(y)
      统一混合：tilde = beta_l * z + (C - beta_l) * Act(z)   # 若无激活，tilde = C * z
      out = x + Dropout(tilde)
    """
    def __init__(self, dim, beta_l: float, C: float, is_last: bool,
                 conv_type='gcn', dropout=0.0, activation=F.relu, norm='layer'):
        super().__init__()
        self.dropout = dropout
        self.activation = activation
        self.is_last = is_last
        self.beta_l = float(beta_l)
        self.C = float(C)
        self.norm = nn.LayerNorm(dim) if norm == 'layer' else nn.BatchNorm1d(dim)
        self.conv = ConvWrapper(conv_type, dim)

    def forward(self, x, edge_index, edge_weight=None):
        y = self.conv(x, edge_index, edge_weight)          # ① 卷积
        z = self.norm(y)                                   # ② 归一化（卷积后）
        if self.activation is None:
            tilde = self.C * z
        else:
            tilde = (self.C - self.beta_l) * self.activation(z) +  self.beta_l * y
        tilde = F.dropout(tilde, p=self.dropout, training=self.training)
        # return tilde + (self.C - self.beta_l) * x  + self.beta_l * z # ④ 残差
        return tilde + (self.C - self.beta_l) * x

class ResNetGNNBase(nn.Module):
    """
    Baseline ResNet-GNN（后激活）：
      x = lin_in(x)
      for L blocks: x = PostActResBlock(x)
      logits = lin_out(x)
    """
    def __init__(self, in_dim, hidden, out_dim, layers=2, dropout=0.5,
                 activation=F.relu, norm='layer', conv_type='gcn'):
        super().__init__()
        assert layers >= 1
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.in_lin = nn.Linear(in_dim, hidden, bias=True)
        self.blocks = nn.ModuleList([
            PostActResBlock(hidden, conv_type=conv_type, dropout=dropout, activation=activation, norm=norm)
            for _ in range(layers)
        ])
        self.out_lin = nn.Linear(hidden, out_dim, bias=True)

    def forward(self, x, edge_index, edge_weight=None, return_hidden=False):
        x = self.in_lin(x)
        for blk in self.blocks:
            x = blk(x, edge_index, edge_weight)
        hidden_feats = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out_lin(x)
        return (x, hidden_feats) if return_hidden else x


class MixResNetGNN(nn.Module):
    """
    Mix 版 ResNet-GNN（后激活）：
      在每个块中先 Conv，再对 Norm(Conv(x)) 进行线性+非线性混合。
    """
    def __init__(self, in_dim, hidden, out_dim, layers=2, dropout=0.5,
                 activation=F.relu, norm='layer', conv_type='gcn',
                 beta: float = 0.5, C: float = 1.0, beta_list: Optional[List[float]] = None):
        super().__init__()
        assert layers >= 1
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.in_lin = nn.Linear(in_dim, hidden, bias=True)

        if beta_list is not None:
            assert len(beta_list) == layers, f"beta_list 长度({len(beta_list)})必须等于 layers({layers})"
            betas = [float(b) for b in beta_list]
        else:
            betas = [float(beta)] * layers

        self.blocks = nn.ModuleList([
            PostActResBlockMix(hidden, beta_l=betas[l], C=C, is_last=(l == layers - 1),
                               conv_type=conv_type, dropout=dropout,
                               activation=activation, norm=norm)
            for l in range(layers)
        ])
        self.out_lin = nn.Linear(hidden, out_dim, bias=True)

    def forward(self, x, edge_index, edge_weight=None, return_hidden=False):
        x = self.in_lin(x)
        for blk in self.blocks:
            x = blk(x, edge_index, edge_weight)
        hidden_feats = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out_lin(x)
        return (x, hidden_feats) if return_hidden else x


# -----------------------------
# 训练/验证/测试
# -----------------------------
@torch.no_grad()
def evaluate(model, data, edge_index, edge_weight, split='val', split_id: int = 0):
    model.eval()
    out = model(data.x, edge_index, edge_weight)           # [N, C]
    pred = out.argmax(dim=-1)
    device = out.device
    if split == 'val':
        mask = _ensure_1d_bool_mask(data.val_mask, split_id, device)
    elif split == 'test':
        mask = _ensure_1d_bool_mask(data.test_mask, split_id, device)
    else:
        mask = _ensure_1d_bool_mask(data.train_mask, split_id, device)

    y = data.y
    if y.device != device:
        y = y.to(device)
    correct = (pred[mask] == y[mask]).sum().item()
    total = int(mask.sum())
    return correct / max(total, 1)


def nll_loss_for_pyg(out, y, mask, split_id: int = 0):
    device = out.device
    mask = _ensure_1d_bool_mask(mask, split_id, device)
    if y.dim() > 1:
        y = y.squeeze(-1)
    if y.device != device:
        y = y.to(device)
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
    # ResNet-GNN 相关
    norm: str = 'layer'            # 'layer' | 'batch'
    conv: str = 'gcn'              # 'gcn' | 'sage' | 'gat'
    # 混合相关
    beta: float = 0.5
    C: float = 1.0
    beta_list: Optional[List[float]] = None
    beta_mode: str = 'custom'      # 'custom' | 'linear' | 'exp' | 'const'
    # 评价相关
    tau: float = 0.95
    abs_target: float = 0.80
    # 数据划分列
    split_id: int = 0
    device: str = 'auto'


def _build_beta_list(layers: int, beta: float, mode: str) -> List[float]:
    if mode == 'custom':
        return []
    if layers <= 1:
        return [beta]
    if mode == 'const':
        return [beta] * layers
    if mode == 'linear':
        beta1, C = 0.95,1
        return [float(beta1 * ((C / beta1) ** (i / (layers - 1))))for i in range(layers)]
    if mode == 'linear1':
        hi, lo = 1,1  # 这里 beta 传 0.3 时，自动线性递减
        return [float(hi - (hi - lo) * l / (layers - 1)) for l in range(layers)]
    raise ValueError(f"Unknown beta_mode: {mode}")


def build_model(cfg: RunConfig, in_dim, out_dim):
    betas = None
    auto_betas = _build_beta_list(cfg.layers, cfg.beta, cfg.beta_mode)
    if auto_betas:
        betas = auto_betas
    if cfg.beta_list is not None:
        betas = cfg.beta_list  # 显式传入优先

    if cfg.model_type == 'baseline':
        return ResNetGNNBase(in_dim, cfg.hidden, out_dim,
                             layers=cfg.layers, dropout=cfg.dropout,
                             activation=F.relu, norm=cfg.norm, conv_type=cfg.conv)
    elif cfg.model_type == 'mix':
        return MixResNetGNN(in_dim, cfg.hidden, out_dim,
                            layers=cfg.layers, dropout=cfg.dropout,
                            activation=F.relu, norm=cfg.norm, conv_type=cfg.conv,
                            beta=cfg.beta, C=cfg.C, beta_list=betas)
    else:
        raise ValueError('Unknown model_type')


# -----------------------------
# 数据加载（五个数据集统一入口）
# -----------------------------
def load_dataset(name: str, root: str, undirected: bool = True):
    name = name.lower()
    if name in ['pubmed']:
        ds = Planetoid(root=os.path.join(root, 'planetoid'), name='PubMed', transform=NormalizeFeatures())
        data = ds[0]
    elif name in ['chameleon', 'squirrel']:
        ds = WikipediaNetwork(root=os.path.join(root, 'wikipedia'),
                              name=name, transform=NormalizeFeatures())
        data = ds[0]
    elif name in ['roman-empire', 'roman_empire', 'romanempire']:
        ds = HeterophilousGraphDataset(root=os.path.join(root, 'hetero'),
                                       name='roman-empire', transform=NormalizeFeatures())
        data = ds[0]
    elif name in ['ogbn-arxiv', 'ogbn_arxiv', 'arxiv']:
        assert OGB_AVAILABLE, "请安装 ogb：pip install ogb"
        ds = PygNodePropPredDataset(
            name='ogbn-arxiv',
            root=os.path.join(root, 'ogbn-arxiv')
        )
        data = ds[0]
        split = ds.get_idx_split()
        N = data.y.size(0)
        train_mask = torch.zeros(N, dtype=torch.bool)
        val_mask   = torch.zeros(N, dtype=torch.bool)
        test_mask  = torch.zeros(N, dtype=torch.bool)
        train_mask[split['train']] = True
        val_mask[split['valid']]   = True
        test_mask[split['test']]   = True
        data.train_mask = train_mask
        data.val_mask   = val_mask
        data.test_mask  = test_mask
        data.y = _ensure_1d_labels(data.y)
        # ogbn-arxiv 常用不做 NormalizeFeatures
    else:
        raise ValueError("dataset 仅支持：Chameleon / Squirrel / Roman-Empire / PubMed / ogbn-arxiv")

    if undirected:
        from torch_geometric.transforms import ToUndirected
        data = ToUndirected()(data)

    edge_index = data.edge_index
    edge_weight = getattr(data, 'edge_weight', None)
    data.y = _ensure_1d_labels(data.y)
    return data, edge_index, edge_weight


# -----------------------------
# 单次训练
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
        loss = nll_loss_for_pyg(out, data.y, data.train_mask, split_id=cfg.split_id)
        loss.backward()

        grad_vec = concat_grads(model)
        grad_norm = float(torch.linalg.norm(grad_vec).item())
        cos = cos_sim(grad_vec, prev_grad_vec) if prev_grad_vec is not None else 0.0
        prev_grad_vec = grad_vec.detach().clone()

        optimizer.step()
        # 🟢 这里对所有 W 做非负+列归一化
        # with torch.no_grad():
        #     for name, p in model.named_parameters():
        #         if "weight" in name and p.dim() == 2:
        #             W_pos = torch.clamp(p, min=0.0)
        #             col_sum = W_pos.sum(dim=0, keepdim=True) + 1e-6
        #             p.copy_(W_pos / col_sum)

        if epoch == cfg.warmup_epochs + 1 and not timer_started:
            now_synchronize()
            wall_start = time.time()
            timer_started = True

        val_acc = evaluate(model, data, edge_index, edge_weight, split='val',  split_id=cfg.split_id)
        test_acc = evaluate(model, data, edge_index, edge_weight, split='test', split_id=cfg.split_id)
        train_acc = evaluate(model, data, edge_index, edge_weight, split='train', split_id=cfg.split_id)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PubMed',
                        choices=['Chameleon','Squirrel','Roman-Empire','PubMed','ogbn-arxiv'])
    parser.add_argument('--layers', type=int, default=32, help='ResNet-GNN 的残差块数量')
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=10)

    # ResNet-GNN 归一化与卷积类型
    parser.add_argument('--norm', type=str, default='layer', choices=['layer','batch'])
    parser.add_argument('--conv', type=str, default='gcn', choices=['gcn','sage','gat'])

    # 混合参数（仅对 mix 生效）
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--beta_mode', type=str, default='custom', choices=['custom', 'linear', 'exp', 'const'],
                        help='自动生成每层 beta（custom 表示使用 --beta_list 或标量 beta）')
    parser.add_argument('--beta_list', type=str, default="1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0")
    parser.add_argument('--C', type=float, default=1)
    

    # 有向/无向
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--undirected', dest='undirected', action='store_true', help='将图统一转为无向')
    group.add_argument('--directed',   dest='undirected', action='store_false', help='保持原始有向（如 ogbn-arxiv）')
    parser.set_defaults(undirected=True)

    # 评价阈值
    parser.add_argument('--tau', type=float, default=0.95)
    parser.add_argument('--abs_target', type=float, default=0.65)

    # 数据划分列（针对 [N,S] 掩码的数据集）
    parser.add_argument('--split_id', type=int, default=0, help='选择多划分掩码的哪一列（默认0）')

    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2])
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

    # 数据加载
    data, edge_index, edge_weight = load_dataset(args.dataset, root=os.path.join(args.outdir, 'data'), undirected=args.undirected)

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
                norm=args.norm,
                conv=args.conv,
                beta=args.beta,
                beta_list=beta_list if model_type == 'mix' else None,
                beta_mode=args.beta_mode,
                C=args.C,
                tau=args.tau,
                abs_target=args.abs_target,
                split_id=args.split_id,
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

    summary_csv = os.path.join(args.outdir, f"summary_{args.dataset}_ResNet_32.csv")
    curves_csv = os.path.join(args.outdir, f"curves_{args.dataset}_ResNet_32.csv")
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
    os.environ["OGB_DISABLE_UPDATE"] = "1"
    main()
