#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从多个 summary_*_relu_L*.csv 中汇总结果：
- 支持多个数据集（Cora / CiteSeer / PubMed）
- 支持多个层数（L3 / L5 / L8 等）
- 对于每个 (dataset, layers, model_type) 单独计算 mean ± std
- 输出表头为：
  model, activation, dataset, type, AULC_epoch_mean, ..., test_at_best_val
  其中 activation 写成 L3, L5, L8 这类形式
"""

import os
import glob
import re
import argparse
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        type=str,
        default="runs/summary_SAGE_*_relu_L*.csv",
        help="匹配 summary csv 文件的 glob 模式，例如 runs/summary_PubMed_relu_L*.csv",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default="runs/agg_by_layers.csv",
        help="汇总结果输出路径",
    )
    args = parser.parse_args()

    # 1. 找到所有 summary 文件
    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"[ERROR] 未找到任何文件，pattern = {args.pattern}")
        return

    print(">>> 将汇总以下文件：")
    for f in files:
        print("  ", f)

    # 2. 读取并拼接所有 summary
    dfs = []
    for f in files:
        df = pd.read_csv(f)

        # 从文件名解析层数，例如 summary_PubMed_relu_L5.csv
        m = re.search(r"_L(\d+)", os.path.basename(f))
        if m:
            L = int(m.group(1))
        else:
            raise ValueError(f"无法从文件名解析层数: {f}")

        df["layers"] = L                # 数值型层数
        df["activation"] = f"L{L}"      # 表格中展示的 activation 列，例如 L3/L5/L8

        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # 3. 需要汇总的指标列（13 个）
    metrics = [
        "AULC_epoch_mean",
        "AULC_epoch_sum",
        "AULC_time",
        "CosSim_mean",
        "E_abs",
        "E_tau",
        "Osc_loss",
        "T_abs",
        "T_tau",
        "VarGrad",
        "best_epoch",
        "best_val",
        "test_at_best_val",
    ]

    # 检查有无缺失的指标列
    missing_cols = [m for m in metrics if m not in df_all.columns]
    if missing_cols:
        print("[Warning] 以下指标在 summary 文件中不存在，将被忽略：", missing_cols)
        metrics = [m for m in metrics if m in df_all.columns]

    rows = []

    # 4. 按 dataset + layers 分开，再对 baseline/mix 分别做 mean±std
    datasets = sorted(df_all["dataset"].unique())

    for ds in datasets:
        df_ds = df_all[df_all["dataset"] == ds]
        for L in sorted(df_ds["layers"].unique()):
            df_L = df_ds[df_ds["layers"] == L]

            for model_type in ["baseline", "mix"]:
                df_m = df_L[df_L["model_type"] == model_type]
                if df_m.empty:
                    continue

                out = {
                    "model": "SAGE",           # 你这批实验都是 GCN，可以按需改成变量
                    "activation": f"L{L}",    # 把层数写到 activation 列
                    "dataset": ds,
                    "type": model_type,       # baseline / mix
                }

                for col in metrics:
                    vals = df_m[col].to_numpy()
                    if len(vals) == 0:
                        out[col] = ""
                    else:
                        mean = float(np.mean(vals))
                        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                        out[col] = f"{mean:.4f} ± {std:.4f}"

                rows.append(out)

    # 5. 保存最终长表
    out_df = pd.DataFrame(rows)

    # 确保目录存在
    out_dir = os.path.dirname(args.save_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_df.to_csv(args.save_csv, index=False, encoding="utf-8")

    print(f"\n>>> 结果已保存到: {args.save_csv}")
    print(out_df)


if __name__ == "__main__":
    main()
