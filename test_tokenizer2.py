# -*- coding: utf-8 -*-
import os
import sys
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from z_any import (
    PCATokenizer,
    get_vel_acc_kappa,
    get_xyyaw_from_polygon_torch,
    transform_polygon_torch_5s,
)

FPS = 5.0

# 处理小yaw跳变问题
def wrap_to_pi(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


# -----------------------------
# action -> traj (x,y,yaw)
# -----------------------------
import numpy as np
import torch


def actions_to_xyyaw_via_polygon_cpu(
    actions_np: np.ndarray,  # [T,3] float
    init_polygons: torch.Tensor,  # [1,4,2] normalized (torch tensor)
    state_range: float,
) -> np.ndarray:
    """
    只在 CPU 上运行：actions -> polygon -> xyyaw
    return: xyyaw_np [T,3] in meters & radians (numpy)
    """
    actions_np = np.asarray(actions_np, dtype=np.float32)
    if actions_np.ndim != 2 or actions_np.shape[1] != 3:
        raise ValueError(f"actions_np must be [T,3], got {actions_np.shape}")

    # 确保 init_poly 在 CPU 且 float32
    init_poly = init_polygons.detach()
    if init_poly.device.type != "cpu":
        init_poly = init_poly.to("cpu")
    if init_poly.dtype != torch.float32:
        init_poly = init_poly.float()

    # [1,T,3] on CPU
    actions = torch.from_numpy(actions_np).unsqueeze(0)  # 默认 CPU，zero-copy view（只要 dtype/contiguous 满足）

    # [1,T,4,2] normalized on CPU
    polys = transform_polygon_torch_5s(init_poly, actions, state_range=state_range)

    # [T,4,2] -> [T,3]
    corners = polys[0]  # [T,4,2]
    xyyaw = get_xyyaw_from_polygon_torch(corners)  # [T,3]

    return xyyaw.detach().numpy()


def get_vx_vy_vyaw(xyyaw: np.ndarray, dt: float = 0.2) -> np.ndarray:
    """
    xyyaw: [T,3] -> return [T,3] = [vx, vy, vyaw]
    vx,vy: global frame 速度；vyaw: yaw 角速度
    """
    xyyaw = np.asarray(xyyaw)
    x = xyyaw[:, 0]
    y = xyyaw[:, 1]
    yaw = xyyaw[:, 2]
    yaw_u = np.unwrap(yaw)  # 关键：消除 -pi/pi 跳变

    vx = np.gradient(x, dt, axis=0)
    vy = np.gradient(y, dt, axis=0)
    vyaw = np.gradient(yaw_u, dt, axis=0)
    return np.stack([vx, vy, vyaw], axis=1)


# -----------------------------
# 新增：对齐 + stack 公共函数
# -----------------------------
def align_and_stack_errors(err_list, time_horizon: int) -> np.ndarray:
    """
    err_list: list of arrays, each expected roughly [Ti, 3]
    return: [N, T, 3] after pad/truncate to time_horizon
    """
    aligned = []
    for idx, err in enumerate(err_list):
        err = np.asarray(err)

        # squeeze 可能出现的多余维度
        if err.ndim == 3:
            err = err.squeeze()
        if err.ndim == 1:
            err = err.reshape(-1, 3)

        if err.ndim != 2 or err.shape[-1] != 3:
            raise ValueError(f"Sample {idx}: expected error shape [T,3], got {err.shape}")

        # 对齐长度：pad 用最后一行，truncate 直接截断
        if err.shape[0] < time_horizon:
            padding = np.tile(err[-1:], (time_horizon - err.shape[0], 1))
            err = np.concatenate([err, padding], axis=0)
        elif err.shape[0] > time_horizon:
            err = err[:time_horizon]

        # 强校验
        if err.shape != (time_horizon, 3):
            # 尝试 reshape / pad / truncate 兜底
            if err.size < time_horizon * 3:
                err_flat = err.flatten()
                pad = time_horizon * 3 - err_flat.size
                err_flat = np.pad(err_flat, (0, pad), mode="constant", constant_values=0)
                err = err_flat.reshape(time_horizon, 3)
            else:
                err = err.flatten()[: time_horizon * 3].reshape(time_horizon, 3)

        if err.shape != (time_horizon, 3):
            raise ValueError(f"Sample {idx}: cannot align to ({time_horizon},3), got {err.shape}")

        aligned.append(err)

    if len(aligned) == 0:
        raise ValueError("No errors collected, dataset might be empty")

    out = np.stack(aligned, axis=0)  # [N,T,3]
    if out.ndim != 3 or out.shape[-1] != 3:
        raise ValueError(f"align_and_stack_errors: expected [N,T,3], got {out.shape}")
    return out


def find_worst_samples(errors):
    var_names = ["dx", "dy", "dyaw"]
    worst_samples = {}

    print(f"当前数据总量: {errors.shape[0]}, 时间步: {errors.shape[1]}, 变量数: {errors.shape[2]}")
    print("-" * 60)

    for i, var in enumerate(var_names):
        data_var = errors[:, :, i]
        max_val_idx = np.argmax(data_var)
        sample_idx, time_step = np.unravel_index(max_val_idx, data_var.shape)

        worst_sample_errors = errors[sample_idx]
        worst_samples[var] = worst_sample_errors

        max_val = data_var[sample_idx, time_step]
        print(f"【{var} 误差最大的样本】")
        print(f"  - 样本索引 (Index): {sample_idx}")
        print(f"  - 发生时间步 (TimeStep): {time_step}")
        print(f"  - 最大误差值: {max_val:.6f}")
        print(f"  - 该样本的完整误差形状: {worst_sample_errors}")
        print("-" * 60)

    return worst_samples


def evaluate_pca_reconstruction(
    dataset,
    tokenizer,
    num_components=15,
    time_horizon=None,
    init_polygons=None,  # torch.Tensor [1,4,2]
    state_range=1.0,  # float
    device="cuda:0",  # str
    dt=0.2,  # float, 用于速度：fps=5
    scenario_name: str = "",  # 场景名（用于打印）
    report_csv_path: str = "./work_dirs/pca_reconstruction_error_report.csv",  # CSV输出路径
    plot_save_path: str = "./work_dirs/pca_errors.png",  # 图输出路径（action误差图）
    subset_original_idxs: Optional[np.ndarray] = None,  # [N]，与 dataset 一一对应
    topk_csv_path: str = "",  # 不为空则输出 TopK 表
    topk: int = 100,
):
    if topk_csv_path:
        if subset_original_idxs is None:
            raise ValueError("topk_csv_path is set, but subset_original_idxs is None")
        if len(subset_original_idxs) != len(dataset):
            raise ValueError(f"subset_original_idxs length mismatch: {len(subset_original_idxs)} vs {len(dataset)}")

    if init_polygons is None:
        raise ValueError("init_polygons is required (torch.Tensor [1,4,2])")

    """
    遍历数据集，计算 PCA 重建误差，并按时间步和变量进行统计分析。
    同时新增：把 action(dx,dy,dyaw) 累计积分为 trajectory(x,y,yaw)，输出 action+traj 共 12 列（max/p99）
    """
    errors = []  # action error |dx,dy,dyaw|
    traj_errors = []  # traj   error |x,y,yaw|
    vel_errors = []  # vel    error |vx,vy,vyaw|
    # 相对误差
    eps = 1e-6
    rel_errors = []
    traj_rel_errors = []
    vel_rel_errors = []
    # 统计列表
    act_sum_list, act_p99_list = [], []
    traj_sum_list, traj_p99_list = [], []
    vel_sum_list, vel_p99_list = [], []

    if len(dataset) > 157172:
        print(dataset[157172])
    if len(dataset) > 280750:
        print(dataset[280750])
    if len(dataset) > 338104:
        print(dataset[338104])

    print(f"开始评估全量数据，样本数: {len(dataset)}，使用组件数: {num_components}...")

    actual_time_horizon = None
    # 循环外：只做一次
    init_poly_cpu = init_polygons.detach().to("cpu", dtype=torch.float32)
    for sample in tqdm(dataset, desc="Encoding/Decoding"):
        sample = np.array(sample)

        tokens = tokenizer(sample, num_components=num_components)
        recon = tokenizer.decode(tokens, num_components=num_components)[0]

        if actual_time_horizon is None:
            print(f"样本形状: {sample.shape}, 解码后形状: {recon.shape}, token数量: {num_components}")

        if actual_time_horizon is None:
            actual_time_horizon = recon.shape[0]
            print(f"检测到时间步数: {actual_time_horizon}")

        min_len = min(sample.shape[0], recon.shape[0])
        if min_len == 0:
            print(f"警告: sample或recon长度为0, sample.shape={sample.shape}, recon.shape={recon.shape}")
            continue

        # -------------------------
        # 1) action 绝对误差--AI写的自定义eps思路
        # -------------------------
        eps_act = np.array([1e-3, 1e-3, 1e-4], dtype=np.float32)  # dx, dy, dyaw
        eps_traj = np.array([1e-2, 1e-2, 1e-4], dtype=np.float32)  # x, y, yaw
        eps_vel = np.array([1e-2, 1e-2, 1e-3], dtype=np.float32)  # vx, vy, vyaw

        gt_act = sample[:min_len]
        re_act = recon[:min_len]

        abs_error = np.abs(gt_act - re_act)
        if abs_error.ndim == 1:
            abs_error = abs_error.reshape(-1, 3)
        errors.append(abs_error)  # abs_error: [T,3]
        act_sum_list.append(abs_error.sum(axis=0))  # [3]
        act_p99_list.append(np.percentile(abs_error, 99, axis=0))  # [3]

        # action 相对误差：|re-gt| / (|gt| + eps)   eps 按维度
        act_rel = np.abs(re_act - gt_act) / (np.abs(gt_act) + eps_act)
        if act_rel.ndim == 1:
            act_rel = act_rel.reshape(-1, 3)
        rel_errors.append(act_rel)

        # -------------------------
        # 2) traj (x,y,yaw) 绝对/相对误差
        # -------------------------
        gt_xyyaw = actions_to_xyyaw_via_polygon_cpu(gt_act, init_poly_cpu, state_range)  # [T,3]
        re_xyyaw = actions_to_xyyaw_via_polygon_cpu(re_act, init_poly_cpu, state_range)  # [T,3]

        # traj_abs_error = np.abs(gt_xyyaw - re_xyyaw) # yaw 通道跳变问题
        diff = re_xyyaw - gt_xyyaw
        diff[:, 2] = wrap_to_pi(diff[:, 2])  # 只处理 yaw 通道
        traj_abs_error = np.abs(diff)

        if traj_abs_error.ndim == 1:
            traj_abs_error = traj_abs_error.reshape(-1, 3)
        traj_errors.append(traj_abs_error)  # traj_abs_error: [T,3]
        traj_sum_list.append(traj_abs_error.sum(axis=0))
        traj_p99_list.append(np.percentile(traj_abs_error, 99, axis=0))
        # traj 相对误差
        traj_rel = np.abs(re_xyyaw - gt_xyyaw) / (np.abs(gt_xyyaw) + eps_traj)
        if traj_rel.ndim == 1:
            traj_rel = traj_rel.reshape(-1, 3)
        traj_rel_errors.append(traj_rel)

        # -------------------------
        # 3) vel (vx,vy,vyaw) 绝对/相对误差
        # -------------------------
        gt_v = get_vx_vy_vyaw(gt_xyyaw, dt=dt)  # [T,3]
        re_v = get_vx_vy_vyaw(re_xyyaw, dt=dt)  # [T,3]

        vel_abs_error = np.abs(gt_v - re_v)
        if vel_abs_error.ndim == 1:
            vel_abs_error = vel_abs_error.reshape(-1, 3)
        vel_errors.append(vel_abs_error)
        vel_sum_list.append(vel_abs_error.sum(axis=0))
        vel_p99_list.append(np.percentile(vel_abs_error, 99, axis=0))
        # vel 相对误差
        vel_rel = np.abs(re_v - gt_v) / (np.abs(gt_v) + eps_vel)
        if vel_rel.ndim == 1:
            vel_rel = vel_rel.reshape(-1, 3)
        vel_rel_errors.append(vel_rel)

    # 推断 time_horizon
    if time_horizon is None:
        if actual_time_horizon is not None:
            time_horizon = actual_time_horizon
        elif len(errors) > 0:
            time_horizon = max(err.shape[0] for err in errors)
            print(f"从误差数组中推断时间步数: {time_horizon}")
        else:
            time_horizon = 14
            print(f"使用默认时间步数: {time_horizon}")

    print(f"使用的时间步数: {time_horizon}")

    # 新增：输出 TopK 表
    if topk_csv_path:
        act_sum = np.stack(act_sum_list, axis=0)  # [N,3]
        act_p99 = np.stack(act_p99_list, axis=0)  # [N,3]
        traj_sum = np.stack(traj_sum_list, axis=0)
        traj_p99 = np.stack(traj_p99_list, axis=0)
        vel_sum = np.stack(vel_sum_list, axis=0)
        vel_p99 = np.stack(vel_p99_list, axis=0)

        K = min(topk, act_sum.shape[0])

        def topk_original_indices(score_1d: np.ndarray) -> np.ndarray:
            # 返回 original_index 的 TopK（降序）
            order = np.argsort(score_1d)[::-1][:K]
            return subset_original_idxs[order]

        # 组装 18 列（每列是一条 TopK 的 original_index 序列）
        cols = {}

        # action: dx dy dyaw
        for j, name in enumerate(["dx", "dy", "dyaw"]):
            cols[f"action_{name}_sum"] = topk_original_indices(act_sum[:, j])
            cols[f"action_{name}_p99"] = topk_original_indices(act_p99[:, j])

        # traj: x y yaw
        for j, name in enumerate(["x", "y", "yaw"]):
            cols[f"traj_{name}_sum"] = topk_original_indices(traj_sum[:, j])
            cols[f"traj_{name}_p99"] = topk_original_indices(traj_p99[:, j])

        # vel: vx vy vyaw
        for j, name in enumerate(["vx", "vy", "vyaw"]):
            cols[f"vel_{name}_sum"] = topk_original_indices(vel_sum[:, j])
            cols[f"vel_{name}_p99"] = topk_original_indices(vel_p99[:, j])

        df_topk = pd.DataFrame(cols)
        os.makedirs(
            os.path.dirname(topk_csv_path) if os.path.dirname(topk_csv_path) else ".",
            exist_ok=True,
        )
        df_topk.to_csv(topk_csv_path, index=False)
        print(f"[TopK] saved: {topk_csv_path}  (K={K})")

    # 对齐 + stack（action + traj 各一份）
    errors = align_and_stack_errors(errors, time_horizon=time_horizon)  # [N,T,3]
    traj_errors = align_and_stack_errors(traj_errors, time_horizon=time_horizon)  # [N,T,3]
    vel_errors = align_and_stack_errors(vel_errors, time_horizon)
    # 相对
    rel_errors = align_and_stack_errors(rel_errors, time_horizon=time_horizon)
    traj_rel_errors = align_and_stack_errors(traj_rel_errors, time_horizon=time_horizon)
    vel_rel_errors = align_and_stack_errors(vel_rel_errors, time_horizon=time_horizon)

    print(f"Errors(action)数组形状: {errors.shape}")
    print(f"Errors(traj)数组形状: {traj_errors.shape}")

    # 基础校验
    if errors.ndim != 3 or errors.shape[2] != 3:
        raise ValueError(f"Expected action errors [N,T,3], got {errors.shape}")
    if traj_errors.ndim != 3 or traj_errors.shape[2] != 3:
        raise ValueError(f"Expected traj errors [N,T,3], got {traj_errors.shape}")

    N, T, C = errors.shape
    print(f"Errors数组: {N}个样本, {T}个时间步, {C}个变量")

    # dyaw / yaw 统一转为 degree
    errors[..., -1] = errors[..., -1] * (180 / np.pi)
    traj_errors[..., -1] = traj_errors[..., -1] * (180 / np.pi)
    vel_errors[..., -1] *= 180 / np.pi

    # -------------------------
    # 统计：action
    # -------------------------
    if N == 1:
        p99_errors = errors[0]
        max_errors = errors[0]
        print("警告: 只有一个样本，action 分位数使用样本本身")
    else:
        p99_errors = np.percentile(errors, 99, axis=0)  # [T,3]
        max_errors = np.max(errors, axis=0)  # [T,3]
    # 统计：action 相对误差
    # -------------------------
    if N == 1:
        rel_p99_errors = rel_errors[0]
        rel_max_errors = rel_errors[0]
    else:
        rel_p99_errors = np.percentile(rel_errors, 99, axis=0)  # [T,3]
        rel_max_errors = np.max(rel_errors, axis=0)  # [T,3]

    # -------------------------
    # 统计：traj（新增）
    # -------------------------
    if N == 1:
        traj_p99_errors = traj_errors[0]
        traj_max_errors = traj_errors[0]
        print("警告: 只有一个样本，traj 分位数使用样本本身")
    else:
        traj_p99_errors = np.percentile(traj_errors, 99, axis=0)  # [T,3]
        traj_max_errors = np.max(traj_errors, axis=0)  # [T,3]
    # traj 相对
    if N == 1:
        traj_rel_p99_errors = traj_rel_errors[0]
        traj_rel_max_errors = traj_rel_errors[0]
    else:
        traj_rel_p99_errors = np.percentile(traj_rel_errors, 99, axis=0)
        traj_rel_max_errors = np.max(traj_rel_errors, axis=0)
    # -------------------------
    # 统计：vel（新增）
    # -------------------------
    if N == 1:
        vel_p99_errors = vel_errors[0]
        vel_max_errors = vel_errors[0]
    else:
        vel_p99_errors = np.percentile(vel_errors, 99, axis=0)  # [T,3]
        vel_max_errors = np.max(vel_errors, axis=0)  # [T,3]
    # vel 相对
    if N == 1:
        vel_rel_p99_errors = vel_rel_errors[0]
        vel_rel_max_errors = vel_rel_errors[0]
    else:
        vel_rel_p99_errors = np.percentile(vel_rel_errors, 99, axis=0)
        vel_rel_max_errors = np.max(vel_rel_errors, axis=0)
    # 变量名
    var_names = ["dx", "dy", "dyaw"]
    traj_names = ["x", "y", "yaw"]
    vel_names = ["vx", "vy", "vyaw"]

    summary_rows = []  # 用于写入 CSV 的 summary 区块

    # 关键结论：误差最大位置（action）
    print("\n" + "=" * 50)
    print("【关键结论：action 误差最大的位置】")
    print("=" * 50)
    for i, var in enumerate(var_names):
        worst_step_max = int(np.argmax(max_errors[:, i])) if T > 1 else 0
        worst_step_p99 = int(np.argmax(p99_errors[:, i])) if T > 1 else 0
        print(f"变量 {var}:")
        print(f"  - [最差情况] 最大误差出现在第 [ {worst_step_max} ] 帧, 误差值: {max_errors[worst_step_max, i].item():.6f}")
        print(f"  - [P99 情况] P99误差出现在第 [ {worst_step_p99} ] 帧, 误差值: {p99_errors[worst_step_p99, i].item():.6f}")
        # summary_rows.append({
        #     "Scenario": scenario_name,
        #     "Section": "action",
        #     "Metric": "max",
        #     "Var": var,
        #     "Frame": worst_step_max,
        #     "Value": float(max_errors[worst_step_max, i]),
        # })
        # summary_rows.append({
        #     "Section": "action",
        #     "Metric": "p99_max_frame",
        #     "Var": var,
        #     "Frame": worst_step_p99,
        #     "Value": float(p99_errors[worst_step_p99, i]),
        # })

    # 关键结论：误差最大位置（traj 新增）
    print("\n" + "=" * 50)
    print("【关键结论：traj 误差最大的位置】")
    print("=" * 50)
    for i, var in enumerate(traj_names):
        worst_step_max = int(np.argmax(traj_max_errors[:, i])) if T > 1 else 0
        worst_step_p99 = int(np.argmax(traj_p99_errors[:, i])) if T > 1 else 0
        print(f"变量 {var}:")
        print(f"  - [最差情况] 最大误差出现在第 [ {worst_step_max} ] 帧, 误差值: {traj_max_errors[worst_step_max, i].item():.6f}")
        print(f"  - [P99 情况] P99误差出现在第 [ {worst_step_p99} ] 帧, 误差值: {traj_p99_errors[worst_step_p99, i].item():.6f}")
        # summary_rows.append({
        #     "Section": "traj",
        #     "Metric": "max",
        #     "Var": var,
        #     "Frame": worst_step_max,
        #     "Value": float(traj_max_errors[worst_step_max, i]),
        # })
        # summary_rows.append({
        #     "Section": "traj",
        #     "Metric": "p99_max_frame",
        #     "Var": var,
        #     "Frame": worst_step_p99,
        #     "Value": float(traj_p99_errors[worst_step_p99, i]),
        # })

    # 关键结论：误差最大速度 (vel 新增）
    print("\n" + "=" * 50)
    print("【关键结论：vel 误差最大的位置】")
    print("=" * 50)
    for i, var in enumerate(vel_names):
        worst_step_max = int(np.argmax(vel_max_errors[:, i])) if T > 1 else 0
        worst_step_p99 = int(np.argmax(vel_p99_errors[:, i])) if T > 1 else 0
        print(f"变量 {var}:")
        print(f"  - [最差情况] 最大误差出现在第 [ {worst_step_max} ] 帧, 误差值: {vel_max_errors[worst_step_max, i].item():.6f}")
        print(f"  - [P99 情况] P99误差出现在第 [ {worst_step_p99} ] 帧, 误差值: {vel_p99_errors[worst_step_p99, i].item():.6f}")
        # summary_rows.append({
        #     "Section": "vel",
        #     "Metric": "max",
        #     "Var": var,
        #     "Frame": worst_step_max,
        #     "Value": float(vel_max_errors[worst_step_max, i]),
        # })
        # summary_rows.append({
        #     "Section": "vel",
        #     "Metric": "p99_max_frame",
        #     "Var": var,
        #     "Frame": worst_step_p99,
        #     "Value": float(vel_p99_errors[worst_step_p99, i]),
        # })

    # 找 worst sample（保留你的原逻辑：对 action）
    _ = find_worst_samples(errors)

    # -------------------------
    # 打印详细表：12列（action 6 + traj 6）
    # -------------------------
    report_data_rel = []
    report_data_abs = []

    for t in range(T):
        row_rel = {"TimeStep": t}
        row_abs = {"TimeStep": t}

        # action
        for i, var in enumerate(var_names):
            row_abs[f"max_{var}"] = max_errors[t, i]
            row_abs[f"{var}_p99"] = p99_errors[t, i]

            row_rel[f"max_{var}"] = rel_max_errors[t, i]
            row_rel[f"{var}_p99"] = rel_p99_errors[t, i]

        # traj
        for i, var in enumerate(traj_names):
            # 绝对误差：维持你现在的列命名规则（_m / _rad）
            if var == "yaw":
                row_abs[f"max_{var}_rad"] = traj_max_errors[t, i]
                row_abs[f"{var}_p99_rad"] = traj_p99_errors[t, i]
            else:
                row_abs[f"max_{var}_m"] = traj_max_errors[t, i]
                row_abs[f"{var}_p99_m"] = traj_p99_errors[t, i]

            # 相对误差：列名保持同一套表头 → 这里也用同样列名写入
            if var == "yaw":
                row_rel[f"max_{var}_rad"] = traj_rel_max_errors[t, i]
                row_rel[f"{var}_p99_rad"] = traj_rel_p99_errors[t, i]
            else:
                row_rel[f"max_{var}_m"] = traj_rel_max_errors[t, i]
                row_rel[f"{var}_p99_m"] = traj_rel_p99_errors[t, i]

        # vel
        for i, var in enumerate(vel_names):
            if var == "vyaw":
                row_abs[f"max_{var}_rad/s"] = vel_max_errors[t, i]
                row_abs[f"{var}_p99_rad/s"] = vel_p99_errors[t, i]
            else:
                row_abs[f"max_{var}_m/s"] = vel_max_errors[t, i]
                row_abs[f"{var}_p99_m/s"] = vel_p99_errors[t, i]

            if var == "vyaw":
                row_rel[f"max_{var}_rad/s"] = vel_rel_max_errors[t, i]
                row_rel[f"{var}_p99_rad/s"] = vel_rel_p99_errors[t, i]
            else:
                row_rel[f"max_{var}_m/s"] = vel_rel_max_errors[t, i]
                row_rel[f"{var}_p99_m/s"] = vel_rel_p99_errors[t, i]

        report_data_rel.append(row_rel)
        report_data_abs.append(row_abs)

    df_rel = pd.DataFrame(report_data_rel)
    df_abs = pd.DataFrame(report_data_abs)

    print("\n" + "=" * 50)
    title = f"【详细数据表 (全部{T}帧) - {scenario_name}】" if scenario_name else f"【详细数据表 (全部{T}帧)】"
    print(title)
    print("写入相对误差块 -> 空行 -> 绝对误差块")
    os.makedirs(
        os.path.dirname(report_csv_path) if os.path.dirname(report_csv_path) else ".",
        exist_ok=True,
    )
    # 先写相对误差（带表头）
    df_rel.to_csv(report_csv_path, index=False, float_format="%.4f")
    # 追加：空一行 + 绝对误差（不写表头）
    with open(report_csv_path, "a") as f:
        f.write("\n")
    df_abs.to_csv(report_csv_path, mode="a", index=False, header=False, float_format="%.4f")
    # 追加 Summary 区块
    # df_summary = pd.DataFrame(summary_rows)
    # with open(report_csv_path, "a") as f:
    #     f.write("\n")          # 空一行
    # df_summary.to_csv(report_csv_path, mode="a", index=False, float_format="%.4f")
    print(f"\n详细数据表已保存至: {report_csv_path}")

    # 可视化：保留 action 原图（避免破坏你现有流程）
    # plot_errors(max_errors, p99_errors, var_names, time_horizon=T)
    plot_errors(max_errors, p99_errors, var_names, time_horizon=T, save_path=plot_save_path)
    # 如需 traj 也画一张，取消注释，并注意保存文件名不要覆盖
    # plot_errors(traj_max_errors, traj_p99_errors, traj_names, time_horizon=T, save_path='./work_dirs/pca_errors_traj.png')

    # 返回保持兼容：仍返回 action 的三项
    return max_errors, p99_errors, var_names


def plot_errors(
    max_errors,
    p99_errors,
    var_names,
    time_horizon=None,
    save_path="./work_dirs/pca_errors.png",
):
    """
    绘制误差随时间步变化的曲线
    """
    if time_horizon is None:
        time_horizon = max_errors.shape[0]
    time_steps = np.arange(time_horizon)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, ax in enumerate(axes):
        ax.plot(time_steps, max_errors[:, i], "b-o", label="max Error", linewidth=2)
        ax.plot(time_steps, p99_errors[:, i], "r--", label="P99 Error", linewidth=1.5)

        ax.set_title(f"{var_names[i]} Reconstruction Error")
        ax.set_xlabel("Time Step (Frame)")
        ax.set_ylabel("Absolute Error")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

        max_idx = int(np.argmax(max_errors[:, i])) if len(time_steps) > 0 else 0
        max_val = max_errors[max_idx, i]
        ax.annotate(
            f"Max: {max_val:.4f}",
            xy=(max_idx, max_val),
            xytext=(max_idx, max_val * 1.1 if max_val != 0 else 0.1),
            arrowprops=dict(facecolor="black", shrink=0.05),
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)


def plot_token_bin_distribution(
    dataset,
    tokenizer,
    num_components=15,
    save_path="./work_dirs/token_bin_distribution.png",
):
    """
    统计并绘制每个token位置的bin分布柱状图
    """
    print(f"开始统计token bin分布，样本数: {len(dataset)}，token数: {num_components}...")

    all_tokens = []
    for sample in tqdm(dataset, desc="Encoding samples"):
        sample = np.array(sample)
        tokens = tokenizer(sample, num_components=num_components)
        all_tokens.append(tokens)

    all_tokens = np.array(all_tokens)
    print(f"Token数组形状: {all_tokens.shape}")

    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    n_rows = 3
    n_cols = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12))
    fig.suptitle(
        f"Token Bin Distribution (Total samples: {len(dataset)}, Vocab size: {vocab_size})",
        fontsize=16,
        fontweight="bold",
    )

    for token_idx in range(num_components):
        row = token_idx // n_cols
        col = token_idx % n_cols
        ax = axes[row, col]

        token_bins = all_tokens[:, token_idx]
        bins, counts = np.unique(token_bins, return_counts=True)

        full_bins = np.arange(vocab_size)
        full_counts = np.zeros(vocab_size, dtype=np.int32)
        for b, c in zip(bins, counts):
            full_counts[b] = c

        ax.bar(
            full_bins,
            full_counts,
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_title(f"Token {token_idx}\n(Used bins: {len(bins)}/{vocab_size})", fontsize=10)
        ax.set_xlabel("Bin ID", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        max_bin = bins[np.argmax(counts)]
        max_count = np.max(counts)
        ax.text(
            0.02,
            0.98,
            f"Max: bin={max_bin}, count={max_count}\nTotal: {len(dataset)}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        if vocab_size > 100:
            used_range = np.where(full_counts > 0)[0]
            if len(used_range) > 0:
                min_bin = max(0, used_range[0] - 10)
                max_bin_show = min(vocab_size - 1, used_range[-1] + 10)
                ax.set_xlim(min_bin, max_bin_show)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nToken bin分布图已保存至: {save_path}")

    print("\n" + "=" * 60)
    print("Token Bin分布统计摘要")
    print("=" * 60)
    for token_idx in range(num_components):
        token_bins = all_tokens[:, token_idx]
        unique_bins = np.unique(token_bins)
        bin_range = f"[{np.min(token_bins)}, {np.max(token_bins)}]"
        used_ratio = len(unique_bins) / vocab_size * 100
        print(
            f"Token {token_idx:2d}: 使用bin数 {len(unique_bins):4d}/{vocab_size:4d} ({used_ratio:5.2f}%), "
            f"范围 {bin_range}, 最频繁bin: {np.argmax(np.bincount(token_bins))}"
        )


if __name__ == "__main__":
    # data_path = './work_dirs/dxdydyaw/all_datas.npy'
    # data_path = "/data-algorithm-hl/zhe.du/planner/plannn2/data_v3/tokenizer/dxdydyaw_pca_20token_1209/all_datas.npy"
    # json_path = './dxdydyaw_pca_20token_1209/dxdydyaw_pca_tokenizer.json'
    json_path = "/share-global/wei.zhang66/work/plannn2/dxdydyaw_pca_20token_1209/dxdydyaw_pca_tokenizer.json"
    device = "cuda:0"
    dt = 0.2
    state_range = 1.0
    init_polygons = torch.tensor(
        [[[-0.0061, -0.0025], [0.0061, -0.0025], [0.0061, 0.0025], [-0.0061, 0.0025]]],
        device=device,
    )
    print("加载数据...")
    # data_info = np.load(data_path, allow_pickle=True).item()
    # # datas = data_info['trajs'][:, :25]
    # datas = data_info['trajs'][:1200, :25]
    # mean_val = data_info['mean_val']
    # std_val = data_info['std_val']
    # print(f"数据形状: {datas.shape}")

    print("加载tokenizer...")
    tokenizer = PCATokenizer(json_path)

    # save_path = './work_dirs/token_bin_distribution_15tokens.png'
    # plot_token_bin_distribution(datas, tokenizer, num_components=15, save_path=save_path)

    # 多场景
    labels_json_path = "/share-global/wei.zhang66/work/plannn2/pca_tokenizer_urban_highway_uturn_aeb_0116/sample_trajectorys_labels.json"
    scenario_npy_path = "/share-global/wei.zhang66/work/plannn2/pca_tokenizer_urban_highway_uturn_aeb_0116/sample_trajectorys_by_scenario.npy"
    scenario_datas_info = np.load(scenario_npy_path, allow_pickle=True).item()
    scenario_datas = scenario_datas_info["trajs"][:, :25]
    print("scenario_datas shape:", scenario_datas.shape)
    with open(labels_json_path, "r") as f:
        meta = json.load(f)
    samples_meta = meta["samples"]  # list of {sampled_index, original_index, category}
    cat2pairs = {}  # cat -> list of (sampled_index, original_index)
    for s in samples_meta:
        cat = s["category"]
        si = int(s["sampled_index"])
        oi = int(s["original_index"])
        cat2pairs.setdefault(cat, []).append((si, oi))
    # 排序，保证可复现
    for cat in cat2pairs:
        cat2pairs[cat].sort(key=lambda x: x[0])  # 按 sampled_index 排序可复现
    cat2idx = {cat: [si for si, oi in pairs] for cat, pairs in cat2pairs.items()}
    print("num categories:", len(cat2idx))

    print("\n开始统计每个主成分系数的重建误差...")

    # max_errors, p99_errors, var_names = evaluate_pca_reconstruction(
    #     datas, tokenizer, num_components=15,
    #     init_polygons=init_polygons,
    #     state_range=state_range,
    #     device=device,
    #     dt=dt,
    # )
    out_dir = "./work_dirs/by_scenario"
    os.makedirs(out_dir, exist_ok=True)

    for cat, pairs in cat2pairs.items():
        sampled_idxs = np.array([p[0] for p in pairs], dtype=np.int64)
        original_idxs = np.array([p[1] for p in pairs], dtype=np.int64)

        # 防御越界：按 sampled_idxs 过滤，同时对 original_idxs 做同样过滤
        mask = (sampled_idxs >= 0) & (sampled_idxs < len(scenario_datas))
        sampled_idxs = sampled_idxs[mask]
        original_idxs = original_idxs[mask]
        if len(sampled_idxs) == 0:
            continue

        subset = scenario_datas[sampled_idxs]  # [Nc,25,3]

        safe_cat = cat.replace("/", "_").replace(" ", "_")
        report_csv = os.path.join(out_dir, f"pca_error_{safe_cat}.csv")
        plot_png = os.path.join(out_dir, f"pca_errors_action_{safe_cat}.png")

        print("\n" + "#" * 80)
        print(f"Running scenario: {cat}  | samples: {len(subset)}")
        print("#" * 80)

        topk_csv = os.path.join(out_dir, f"top100_indices_{safe_cat}.csv")
        evaluate_pca_reconstruction(
            subset,
            tokenizer,
            num_components=15,
            init_polygons=init_polygons,
            state_range=state_range,
            device=device,
            dt=dt,
            scenario_name=cat,
            report_csv_path=report_csv,  # 原来的曲线表
            plot_save_path=plot_png,
            subset_original_idxs=original_idxs,  # 新增
            topk_csv_path=topk_csv,  # 新增
            topk=100,  # 新增
        )
# nohup python test_tokenizer2.py   > test_tokenizer2.log 2>&1 &
# plot_errors(max_errors, p99_errors, var_names)
