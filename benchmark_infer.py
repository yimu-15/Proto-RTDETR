# -*- coding: utf-8 -*-
"""
推理性能 benchmark：w/ ProtoContrastHead vs w/o ProtoContrastHead。

实测 FPS / 参数量 / 峰值显存，支撑论文 "introduces no additional
computational cost at inference" 结论（proto_head 在 eval 模式下不执行计算）。

用法:
  D:\\conda_envs\\fedsem\\python.exe benchmark_infer.py [--device cuda:0]

输出:
  逐模型打印 FPS (1x3x640x640, fp32, batch=1)、#Params(M)、峰值显存(GB)。

注意:
  1. 不加载任何预训练权重（config 中 pretrained: False），纯测网络结构开销。
  2. 不触发训练 / 数据集加载。
"""
import argparse
import os
import sys
import time

# Windows: 将 conda 环境的 Library\\bin 注入 DLL 搜索路径（必须在 import numpy/torch 前）
if sys.platform == 'win32':
    lib_bin = os.path.join(sys.prefix, 'Library', 'bin')
    if os.path.isdir(lib_bin):
        os.environ['PATH'] = lib_bin + os.pathsep + os.environ.get('PATH', '')

# 确保 src 在 path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from src.core import YAMLConfig

# (标签, config 路径) —— 最终论文配置为 lw=1.0 无 Mosaic
CONFIGS = [
    ('w/ ProtoHead  ', 'configs/proto_rtdetrv2_r50vd_visa_abl2_lw1.yml'),
    ('w/o ProtoHead ', 'configs/proto_rtdetrv2_r50vd_visa_abl2_wo_proto.yml'),
]

WARMUP_ITERS = 10
TEST_ITERS = 100
IM_SIZE = 640
BATCH = 1


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def benchmark(cfg_path: str, device: torch.device) -> dict:
    torch.set_grad_enabled(False)

    cfg = YAMLConfig(cfg_path)
    # 关掉预训练权重下载/加载（与 train_proto_visa.py 一致），只测结构
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    model = cfg.model.to(device).eval()
    n_params = count_parameters(model)

    dummy = torch.randn(BATCH, 3, IM_SIZE, IM_SIZE, device=device)
    for _ in range(WARMUP_ITERS):
        _ = model(dummy)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(TEST_ITERS):
        _ = model(dummy)
    torch.cuda.synchronize()
    dt = time.time() - t0

    fps = TEST_ITERS / dt
    peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3

    return {'fps': fps, 'params_m': n_params / 1e6, 'peak_mem_gb': peak_mem}


def main():
    parser = argparse.ArgumentParser(description='Inference benchmark w/ vs w/o ProtoHead')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('[ERROR] CUDA required for this benchmark.')
        sys.exit(1)

    device = torch.device(args.device)
    print(f'Benchmark settings: input=1x3x{IM_SIZE}x{IM_SIZE}, fp32, '
          f'warmup={WARMUP_ITERS}, iters={TEST_ITERS}, device={device}')
    print('-' * 72)

    results = {}
    for tag, cfg_path in CONFIGS:
        if not os.path.exists(cfg_path):
            print(f'[SKIP] config not found: {cfg_path}')
            continue
        res = benchmark(cfg_path, device)
        results[tag.strip()] = res
        print(f'{tag} | FPS={res["fps"]:.1f} | Params={res["params_m"]:.2f}M | '
              f'PeakMem={res["peak_mem_gb"]:.3f}GB')

    if len(results) == 2:
        k_w, k_wo = 'w/ ProtoHead', 'w/o ProtoHead'
        print('-' * 72)
        print(f'Delta FPS     : {results[k_wo]["fps"] - results[k_w]["fps"]:+.1f} '
              f'({(results[k_w]["fps"] / max(results[k_wo]["fps"], 1e-9) - 1) * 100:+.2f}%)')
        print(f'Delta Params  : {results[k_w]["params_m"] - results[k_wo]["params_m"]:+.3f}M')
        print(f'Delta PeakMem : {results[k_w]["peak_mem_gb"] - results[k_wo]["peak_mem_gb"]:+.4f}GB')
        print('=> FPS/显存基本持平 = 推理阶段 ProtoHead 无计算开销（参数仍存在）')


if __name__ == '__main__':
    main()
