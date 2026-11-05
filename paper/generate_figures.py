# -*- coding: utf-8 -*-
"""
从 output/*/log.txt 提取 per-epoch 指标，生成论文图片。

生成的图（输出到 paper/figures/）：
  - fig1_mAP_curves.png    : val mAP@[.5:.95] 训练曲线（含 Mosaic / 无 Mosaic 两个 regime）
  - fig1_train_loss.png    : train 总 loss 曲线（同上两个 regime）
  - fig2_ablation_bars.png : 两组消融的 best mAP 条形图

用法：
  D:\\conda_envs\\fedsem\\python.exe paper/generate_figures.py

注意：Windows 下 conda 环境不激活时，`<env>\\Library\\bin` 不在 PATH 中，
numpy 延迟加载 mkl_rt.3.dll 会失败（0xc06d007f），因此脚本启动时自动注入。
"""
import json
import os
import sys

# Windows: 将 conda 环境的 Library\\bin 注入 DLL 搜索路径（必须在 import numpy/matplotlib 前）
if sys.platform == 'win32':
    lib_bin = os.path.join(sys.prefix, 'Library', 'bin')
    if os.path.isdir(lib_bin):
        os.environ['PATH'] = lib_bin + os.pathsep + os.environ.get('PATH', '')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
OUT = os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)

# (exp_dir, 图例名, 颜色, 标记)
REGIME_A = [  # 标准增强（含 Mosaic）
    ('proto_rtdetrv2_r50vd_visa_abl_wo_proto', 'w/o ProtoHead',
     '#9e9e9e', 'o'),
    ('proto_rtdetrv2_r50vd_visa_v2',            'w/ ProtoHead ($\\lambda$=0.5)',
     '#1976d2', 's'),
    ('proto_rtdetrv2_r50vd_visa_abl_lw1',       'w/ ProtoHead ($\\lambda$=1.0)',
     '#d32f2f', '^'),
]
REGIME_B = [  # 无 Mosaic
    ('proto_rtdetrv2_r50vd_visa_abl2_wo_proto', 'w/o ProtoHead',
     '#9e9e9e', 'o'),
    ('proto_rtdetrv2_r50vd_visa_abl_no_mosaic', 'w/ ProtoHead ($\\lambda$=0.5)',
     '#1976d2', 's'),
    ('proto_rtdetrv2_r50vd_visa_abl2_lw1',      'w/ ProtoHead ($\\lambda$=1.0)',
     '#d32f2f', '^'),
]

REGIME_A_TITLE = 'Standard augmentation (with Mosaic)'
REGIME_B_TITLE = 'Mosaic-free augmentation'


def load_exp(exp_dir):
    """读取 log.txt，返回按 epoch 排序的指标列表（同一 epoch 保留最后一行）。"""
    log_path = os.path.join(ROOT, 'output', exp_dir, 'log.txt')
    rows = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'test_coco_eval_bbox' not in d:
                continue
            st = d['test_coco_eval_bbox']
            rows.append({
                'epoch': int(d['epoch']),
                'train_loss': d.get('train_loss'),
                'mAP': st[0],
                'AP50': st[1],
                'AP75': st[2],
                'AR100': st[8],
            })
    dedup = {}
    for r in rows:
        dedup[r['epoch']] = r  # 同一 epoch 后写覆盖
    return [dedup[k] for k in sorted(dedup)]


def best_row(rows):
    """返回 best mAP 所在行（及 best 值）。"""
    br = max(rows, key=lambda r: r['mAP'])
    return br, br['mAP']


def draw_curves(ax, regime, metric, ylabel, title, ylim=None):
    for exp, name, color, marker in regime:
        rows = load_exp(exp)
        if not rows:
            print(f'[WARN] {exp}: no data')
            continue
        xs = [r['epoch'] for r in rows]
        ys = [r[metric] for r in rows]
        ax.plot(xs, ys, color=color, marker=marker, ms=4, lw=1.6, label=name)
        br, _ = best_row(rows)
        ax.plot(br['epoch'], br['mAP'], color=color, marker='*', ms=12,
                zorder=5, ls='none')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)


def draw_bars(ax, regime, title, ylim=None):
    names = [n for _, n, _, _ in regime]
    vals = []
    for exp, _, _, _ in regime:
        rows = load_exp(exp)
        _, b = best_row(rows) if rows else (None, float('nan'))
        vals.append(b)
    colors = [c for _, _, c, _ in regime]
    bars = ax.bar(names, vals, color=colors, width=0.62, edgecolor='black',
                  linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('best mAP@[.5:.95]')
    ax.set_title(title, fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)


# ========== Figure 1a: val mAP 曲线 ==========
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
draw_curves(axes[0], REGIME_A, 'mAP', 'val mAP@[.5:.95]',
            REGIME_A_TITLE, ylim=(0.15, 0.40))
draw_curves(axes[1], REGIME_B, 'mAP', 'val mAP@[.5:.95]',
            REGIME_B_TITLE, ylim=(0.80, 0.95))
fig.suptitle('Validation mAP@[.5:.95] over training', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, 'fig1_mAP_curves.png'), dpi=200)
plt.close(fig)
print('saved fig1_mAP_curves.png')

# ========== Figure 1b: train loss 曲线 ==========
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
draw_curves(axes[0], REGIME_A, 'train_loss', 'train total loss',
            REGIME_A_TITLE)
draw_curves(axes[1], REGIME_B, 'train_loss', 'train total loss',
            REGIME_B_TITLE)
fig.suptitle('Training total loss over epochs', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, 'fig1_train_loss.png'), dpi=200)
plt.close(fig)
print('saved fig1_train_loss.png')

# ========== Figure 2: 消融条形图 ==========
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
draw_bars(axes[0], REGIME_A, REGIME_A_TITLE, ylim=(0.20, 0.42))
draw_bars(axes[1], REGIME_B, REGIME_B_TITLE, ylim=(0.80, 0.96))
fig.suptitle('Ablation of ProtoContrastHead (best mAP)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, 'fig2_ablation_bars.png'), dpi=200)
plt.close(fig)
print('saved fig2_ablation_bars.png')

# ========== Figure 3: τ 温度敏感性折线图 ==========
TAU_RUNS = [  # (exp_dir, τ, 颜色, 标记)
    ('proto_rtdetrv2_r50vd_visa_abl2_lw1_tau002', 0.02, '#1976d2', 'o'),
    ('proto_rtdetrv2_r50vd_visa_abl2_lw1_tau007', 0.07, '#d32f2f', 's'),
    ('proto_rtdetrv2_r50vd_visa_abl2_lw1_tau01',  0.10, '#f9a825', '^'),
    ('proto_rtdetrv2_r50vd_visa_abl2_lw1_tau03',  0.30, '#7b1fa2', 'v'),
]

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

# 左面板：6-epoch 预算下 per-epoch val mAP 曲线
for exp, tau, color, marker in TAU_RUNS:
    rows = load_exp(exp)
    if not rows:
        print(f'[WARN] {exp}: no data')
        continue
    xs = [r['epoch'] for r in rows]
    ys = [r['mAP'] for r in rows]
    axes[0].plot(xs, ys, color=color, marker=marker, ms=4.5, lw=1.6,
                 label=f'$\\tau$={tau:.2f}')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('val mAP@[.5:.95]')
axes[0].set_title('Per-epoch curves (6-epoch budget)', fontsize=11)
axes[0].legend(frameon=False, fontsize=8)
axes[0].grid(alpha=0.3)
axes[0].set_ylim(0.0, 0.40)

# 右面板：best / final mAP vs τ（对数 x 轴），标注数值
taus = [t for _, t, _, _ in TAU_RUNS]
best_vals, final_vals = [], []
for exp, _, _, _ in TAU_RUNS:
    rows = load_exp(exp)
    _, b = best_row(rows) if rows else (None, float('nan'))
    best_vals.append(b)
    final_vals.append(rows[-1]['mAP'] if rows else float('nan'))

axes[1].plot(taus, best_vals, color='#1976d2', marker='o', ms=5, lw=1.8,
             ls='-', label='best mAP (within budget)')
axes[1].plot(taus, final_vals, color='#d32f2f', marker='^', ms=5, lw=1.6,
             ls='--', label='final mAP@epoch 6')
for x, y_b, y_f in zip(taus, best_vals, final_vals):
    axes[1].annotate(f'{y_b:.3f}', (x, y_b), textcoords='offset points',
                     xytext=(0, 7), ha='center', fontsize=8, color='#1976d2')
    axes[1].annotate(f'{y_f:.3f}', (x, y_f), textcoords='offset points',
                     xytext=(0, -13), ha='center', fontsize=8, color='#d32f2f')
axes[1].set_xscale('log')
axes[1].set_xticks(taus)
axes[1].set_xticklabels([f'{t:.2f}' for t in taus])
axes[1].set_xlabel('Temperature $\\tau$ (log scale)')
axes[1].set_ylabel('mAP@[.5:.95]')
axes[1].set_title('Temperature sensitivity (6-epoch budget)', fontsize=11)
axes[1].legend(frameon=False, fontsize=8)
axes[1].grid(alpha=0.3, which='both')
axes[1].set_ylim(0.15, 0.40)

fig.suptitle('Temperature sensitivity of the InfoNCE head', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, 'fig3_tau_sensitivity.png'), dpi=200)
plt.close(fig)
print('saved fig3_tau_sensitivity.png')

# ========== 控制台汇总（便于核对） ==========
print('\n===== 汇总（best mAP / best epoch）=====')
for label, regime in [('Regime A (with Mosaic)', REGIME_A),
                      ('Regime B (no Mosaic)', REGIME_B)]:
    print(f'\n{label}:')
    for exp, name, _, _ in regime:
        rows = load_exp(exp)
        if not rows:
            print(f'  {name}: NO DATA')
            continue
        br, b = best_row(rows)
        print(f'  {name:<28} {b:.3f} @ epoch {br["epoch"]}'
              f'  (AP50={br["AP50"]:.3f}, AP75={br["AP75"]:.3f}, AR100={br["AR100"]:.3f})')

print('\n===== τ 温度消融（per-epoch mAP）=====')
for exp, tau, _, _ in TAU_RUNS:
    rows = load_exp(exp)
    print(f'\n  tau={tau:.2f} ({exp}):')
    for r in rows:
        print(f'    E{r["epoch"]}: mAP={r["mAP"]:.3f} AP50={r["AP50"]:.3f}'
              f' AP75={r["AP75"]:.3f} AR100={r["AR100"]:.3f}')
    br, b = best_row(rows)
    print(f'    best={b:.3f} @ E{br["epoch"]}'
          f' | final E{rows[-1]["epoch"]}={rows[-1]["mAP"]:.3f}')
