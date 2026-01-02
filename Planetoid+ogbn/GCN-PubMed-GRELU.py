
# gnn_compare_stability_speed.py
# 比较 Baseline-GCN 与 MixGCN 的收敛速度与稳定性（多指标，多随机种子）
import os, time, math, argparse, random, warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm


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
# 说明：Planetoid 通常以无向图的“双向边”存储；0.5 系数在双向边时能抵消双计数。
# 若不是双向存储，0.5 会让整体数值缩放为一半——但本指标仅用于“相对比较”，不影响结论。
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
# 自定义激活：GReLU
# 形式：
#   x < 0         -> a * x
#   0 <= x < c    -> b * x
#   x >= c        -> d * x
# 采用共享标量参数（对所有通道一致），更稳定也更省参数。
# 如需通道级参数，可扩展为向量并注意广播与阈值比较。
# -----------------------------
class GReLU(nn.Module):
    def __init__(self, init_a: float = 0.1, init_b: float = 1.0,
                 init_c: float = 1.0, init_d: float = 0.5,
                 learnable: bool = True):
        super().__init__()
        if learnable:
            self.a = nn.Parameter(torch.tensor(init_a))
            self.b = nn.Parameter(torch.tensor(init_b))
            self.c = nn.Parameter(torch.tensor(init_c))
            self.d = nn.Parameter(torch.tensor(init_d))
        else:
            self.register_buffer("a", torch.tensor(init_a))
            self.register_buffer("b", torch.tensor(init_b))
            self.register_buffer("c", torch.tensor(init_c))
            self.register_buffer("d", torch.tensor(init_d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 注意：where 的条件用 x 和标量 c 比较，能自然广播
        out = torch.where(x < 0, self.a * x, x)                    # x<0
        out = torch.where((x >= 0) & (x < self.c), self.b * x, out) # 0<=x<c
        out = torch.where(x >= self.c, self.d * x, out)             # x>=c
        return out


# -----------------------------
# 模型定义：Baseline 与 MixGCN（GReLU 版）
# -----------------------------
class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 dropout: float = 0.5,
                 use_grelu: bool = True):
        super().__init__()
        # 预归一化已在外部完成 -> normalize=False；cached=False 避免无意义缓存
        self.conv = GCNConv(in_dim, out_dim, cached=False, normalize=False, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.activation = GReLU() if use_grelu else None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.dropout(x)
        h = self.conv(x, edge_index, edge_weight)
        return self.activation(h) if self.activation is not None else h


class MixGCNLayer(nn.Module):
    """
    X_{l+1} = beta * Z + (C - beta) * sigma(Z),  Z = \tilde{A} X_l W_l
    """
    def __init__(self, in_dim: int, out_dim: int,
                 beta: float = 0.5, C: float = 1.0,
                 dropout: float = 0.5,
                 use_grelu: bool = True):
        super().__init__()
        self.conv = GCNConv(in_dim, out_dim, cached=False, normalize=False, bias=True)
        self.beta = float(beta)
        self.C = float(C)
        self.dropout = nn.Dropout(dropout)
        self.activation = GReLU() if use_grelu else None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.dropout(x)
        z = self.conv(x, edge_index, edge_weight)
        if self.activation is None:
            # 最后一层：C * z（不改变 argmax，只做缩放）
            return self.C * z
        return self.beta * z + (self.C - self.beta) * self.activation(z)


class BaselineGCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 dropout: float = 0.5, layers: int = 2):
        super().__init__()
        assert layers >= 2
        mods = []
        dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        for l in range(layers):
            idim, odim = dims[l], dims[l + 1]
            # 最后一层不用激活
            use_act = (l < layers - 1)
            mods.append(GCNLayer(idim, odim, dropout=dropout, use_grelu=use_act))
        self.layers = nn.ModuleList(mods)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor],
                return_hidden: bool = False):
        hidden_feats = None
        for l, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            # 倒数第二层输出，作为过平滑监控的特征
            if l == len(self.layers) - 2:
                hidden_feats = x
        return (x, hidden_feats) if return_hidden else x


class MixGCN(nn.Module):
    """
    支持每层不同的 beta：
      - 若传入 beta_list（长度==layers），逐层使用；
      - 否则使用标量 beta（各层相同）。
    C 仍为全局标量。
    """
    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 dropout: float = 0.5, layers: int = 2,
                 beta: float = 0.5, C: float = 1.0,
                 beta_list: Optional[List[float]] = None):
        super().__init__()
        assert layers >= 2
        dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]

        if beta_list is not None:
            assert len(beta_list) == layers, f"beta_list 长度({len(beta_list)})必须等于 layers({layers})"

        mods = []
        for l in range(layers):
            idim, odim = dims[l], dims[l + 1]
            use_act = (l < layers - 1)  # 最后一层不做激活
            beta_l = beta_list[l] if beta_list is not None else beta
            mods.append(MixGCNLayer(idim, odim,
                                    beta=beta_l,
                                    C=C,
                                    dropout=dropout,
                                    use_grelu=use_act))
        self.layers = nn.ModuleList(mods)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor],
                return_hidden: bool = False):
        hidden_feats = None
        for l, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            if l == len(self.layers) - 2:
                hidden_feats = x
        return (x, hidden_feats) if return_hidden else x

# -----------------------------
# 训练/验证/测试（删掉无用二分类 fallback）
# -----------------------------
@torch.no_grad()
def evaluate(model, data, edge_index, edge_weight, split='val'):
    model.eval()
    out = model(data.x, edge_index, edge_weight)           # [N, C]
    pred = out.argmax(dim=-1)                              # Planetoid 均为多分类
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
    beta_mode: str = 'custom'  # 新增：'custom' | 'linear' | 'exp' | 'const'


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
        return BaselineGCN(in_dim, cfg.hidden, out_dim, dropout=cfg.dropout, layers=cfg.layers)
    elif cfg.model_type == 'mix':
        return MixGCN(in_dim, cfg.hidden, out_dim, dropout=cfg.dropout, layers=cfg.layers,
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

    # 记录曲线
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

        # warmup 后开始计时（GPU 同步；CPU 下无需同步，wall_time 仍可用）
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

        # 早停
        if val_acc > best_val + 1e-12:
            best_val = val_acc
            best_test_at_val = test_acc
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= cfg.patience:
            break

    # 计算速度与稳定性指标（基于曲线）
    df = pd.DataFrame(rows)

    # AULC：给出“按 epoch 的均值”（可比）与“按时间”的积分
    aulc_epoch_sum = df['val_acc'].sum()                 # 原始 sum（仅作参考）
    aulc_epoch_mean = float(df['val_acc'].mean())        # ✅ 可比
    t = df['wall_time'].to_numpy()
    a = df['val_acc'].to_numpy()
    aulc_time = float(np.trapz(a, t)) if len(df) >= 2 else 0.0

    # 绝对阈值 & 相对最优命中
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

    # 梯度稳定性
    grad_var = float(df['grad_norm'].var()) if len(df) > 1 else 0.0
    cos_mean = float(df['grad_cos'].iloc[1:].mean()) if len(df) > 2 else 0.0  # 第1个无前项
    # 损失震荡度（相对变化的均值）
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
    parser.add_argument('--dataset', type=str, default='PubMed', choices=['Cora', 'CiteSeer', 'PubMed'])
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--warmup_epochs', type=int, default=5)

    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--beta_mode', type=str, default='custom', choices=['custom', 'linear', 'exp', 'const'],
                        help='自动生成每层 beta（custom 表示使用 --beta_list 或标量 beta）')
    parser.add_argument('--beta_list', type=str, default="0.98,0.95,0.90")
    parser.add_argument('--C', type=float, default=1.0)

    parser.add_argument('--tau', type=float, default=0.95)
    parser.add_argument('--abs_target', type=float, default=0.6)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9])
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

    # 数据与图归一化（共享）
    dataset = Planetoid(root=os.path.join(args.outdir, 'data'), name=args.dataset, transform=NormalizeFeatures())
    data = dataset[0]
    edge_index, edge_weight = gcn_norm(
        data.edge_index, data.edge_weight, num_nodes=data.num_nodes, add_self_loops=True
    )

    # 一次性打印边对称性（用于 Dirichlet 解释）
    inspect_edge_symmetry(edge_index, num_nodes=data.num_nodes, verbose=True)

    all_summaries = []
    all_curves = []

    # 两个模型：baseline & mix
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

    summary_csv = os.path.join(args.outdir, f"summary_{args.dataset}_GRELU.csv")
    curves_csv = os.path.join(args.outdir, f"curves_{args.dataset}_GRELU.csv")
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
