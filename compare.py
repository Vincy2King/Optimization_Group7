#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化器性能比较脚本
比较 GD (Gradient Descent), SGD (Stochastic Gradient Descent), 
NAG (Nesterov Accelerated Gradient), Adam 之间的性能

作者: Auto-generated
日期: 2024
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
from sklearn.datasets import load_svmlight_file

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 导入各个优化器模块
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from GD import gradient_descent, calculate_cost, load_a9a_data
from SGD import sgd_logreg, compute_loss as sgd_compute_loss
from NAG import nesterov_accelerated_gradient, compute_loss as nag_compute_loss
from Adam import train_adam, logistic_loss_with_l2


def load_unified_data(data_dir: str = "data") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    统一的数据加载函数，确保所有优化器使用相同的数据集。
    
    参数:
        data_dir: 数据目录路径
    
    返回:
        X_train: 训练特征矩阵 (n_train_samples, n_features)
        y_train: 训练标签向量 {-1, 1} (n_train_samples,)
        X_test: 测试特征矩阵 (n_test_samples, n_features)
        y_test: 测试标签向量 {-1, 1} (n_test_samples,)
    
    异常:
        FileNotFoundError: 当数据文件不存在时抛出
    """
    # 尝试多个可能的路径
    possible_train_paths = [
        os.path.join(data_dir, "a9a"),
        os.path.join(".", "a9a"),
        os.path.join(".", "a9a.dat"),
        "a9a",
        "a9a.dat"
    ]
    
    possible_test_paths = [
        os.path.join(data_dir, "a9a.t"),
        os.path.join(".", "a9a.t"),
        os.path.join(".", "a9at.dat"),
        "a9a.t",
        "a9at.dat"
    ]
    
    train_path = None
    test_path = None
    
    for path in possible_train_paths:
        if os.path.exists(path):
            train_path = path
            break
    
    for path in possible_test_paths:
        if os.path.exists(path):
            test_path = path
            break
    
    if train_path is None or test_path is None:
        raise FileNotFoundError(
            f"数据文件未找到。请确保以下文件存在:\n"
            f"  训练集: {data_dir}/a9a 或 ./a9a 或 ./a9a.dat\n"
            f"  测试集: {data_dir}/a9a.t 或 ./a9a.t 或 ./a9at.dat"
        )
    
    n_features = 123
    try:
        X_train_sparse, y_train = load_svmlight_file(train_path, n_features=n_features)
        X_test_sparse, y_test = load_svmlight_file(test_path, n_features=n_features)
    except Exception as e:
        raise FileNotFoundError(f"加载数据文件失败: {e}")
    
    X_train = X_train_sparse.toarray()
    X_test = X_test_sparse.toarray()
    
    # 添加偏置项
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    
    # 确保标签为 {-1, 1}
    y_train = y_train.astype(np.float64)
    y_test = y_test.astype(np.float64)
    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1
    
    print(f"数据加载完成: 训练集 {X_train.shape}, 测试集 {X_test.shape}")
    return X_train, y_train, X_test, y_test


def run_gd_optimizer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lambda_reg: float,
    learning_rate: float,
    max_iters: int
) -> Dict[str, any]:
    """
    运行 GD 优化器。
    
    参数:
        X_train: 训练特征矩阵
        y_train: 训练标签向量
        X_test: 测试特征矩阵
        y_test: 测试标签向量
        lambda_reg: 正则化系数
        learning_rate: 学习率
        max_iters: 最大迭代次数
    
    返回:
        包含训练结果的字典
    """
    print("\n" + "="*60)
    print("运行 GD (Gradient Descent) 优化器...")
    print("="*60)
    
    start_time = time.time()
    w, train_loss, test_loss, test_acc, grad_norm = gradient_descent(
        X=X_train,
        y=y_train,
        lambda_val=lambda_reg,
        learning_rate=learning_rate,
        max_iters=max_iters,
        X_val=X_test,
        y_val=y_test
    )
    elapsed_time = time.time() - start_time
    
    return {
        "name": "GD",
        "weights": w,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "grad_norm": grad_norm,
        "time": elapsed_time,
        "iterations": max_iters
    }


def run_sgd_optimizer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lambda_reg: float,
    learning_rate: float,
    n_epochs: int,
    batch_size: int = 256
) -> Dict[str, any]:
    """
    运行 SGD 优化器。
    
    参数:
        X_train: 训练特征矩阵
        y_train: 训练标签向量
        X_test: 测试特征矩阵
        y_test: 测试标签向量
        lambda_reg: 正则化系数
        learning_rate: 学习率
        n_epochs: 训练轮数
        batch_size: 批次大小
    
    返回:
        包含训练结果的字典
    """
    print("\n" + "="*60)
    print("运行 SGD (Stochastic Gradient Descent) 优化器...")
    print("="*60)
    
    start_time = time.time()
    w, train_loss, test_loss, test_acc, grad_norm = sgd_logreg(
        X=X_train,
        t=y_train,
        lambda_reg=lambda_reg,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        batch_size=batch_size,
        X_test=X_test,
        y_test=y_test,
        random_state=42
    )
    elapsed_time = time.time() - start_time
    
    return {
        "name": "SGD",
        "weights": w,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "grad_norm": grad_norm,
        "time": elapsed_time,
        "iterations": n_epochs
    }


def run_nag_optimizer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lambda_reg: float,
    learning_rate: float,
    n_iterations: int,
    momentum_beta: float = 0.9
) -> Dict[str, any]:
    """
    运行 NAG 优化器。
    
    参数:
        X_train: 训练特征矩阵
        y_train: 训练标签向量
        X_test: 测试特征矩阵
        y_test: 测试标签向量
        lambda_reg: 正则化系数
        learning_rate: 学习率
        n_iterations: 迭代次数
        momentum_beta: 动量参数
    
    返回:
        包含训练结果的字典
    """
    print("\n" + "="*60)
    print("运行 NAG (Nesterov Accelerated Gradient) 优化器...")
    print("="*60)
    
    start_time = time.time()
    w, train_loss, test_loss, test_acc, grad_norm = nesterov_accelerated_gradient(
        X=X_train,
        t=y_train,
        lambda_reg=lambda_reg,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        momentum_beta=momentum_beta,
        X_test=X_test,
        y_test=y_test
    )
    elapsed_time = time.time() - start_time
    
    return {
        "name": "NAG",
        "weights": w,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "grad_norm": grad_norm,
        "time": elapsed_time,
        "iterations": n_iterations
    }


def run_adam_optimizer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lambda_reg: float,
    learning_rate: float,
    n_iterations: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8
) -> Dict[str, any]:
    """
    运行 Adam 优化器。
    
    参数:
        X_train: 训练特征矩阵
        y_train: 训练标签向量
        X_test: 测试特征矩阵
        y_test: 测试标签向量
        lambda_reg: 正则化系数
        learning_rate: 学习率
        n_iterations: 迭代次数
        beta1: Adam 的 beta1 参数
        beta2: Adam 的 beta2 参数
        eps: Adam 的 epsilon 参数
    
    返回:
        包含训练结果的字典
    """
    print("\n" + "="*60)
    print("运行 Adam 优化器...")
    print("="*60)
    
    start_time = time.time()
    w, train_loss, test_loss, test_acc, grad_norm = train_adam(
        Xtr=X_train,
        ytr=y_train,
        Xte=X_test,
        yte=y_test,
        lam=lambda_reg,
        lr=learning_rate,
        iters=n_iterations,
        beta1=beta1,
        beta2=beta2,
        eps=eps
    )
    elapsed_time = time.time() - start_time
    
    return {
        "name": "Adam",
        "weights": w,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "grad_norm": grad_norm,
        "time": elapsed_time,
        "iterations": n_iterations
    }


def visualize_comparison(results: List[Dict[str, any]], output_dir: str = ".") -> None:
    """
    可视化比较所有优化器的性能。
    
    参数:
        results: 所有优化器的结果列表
        output_dir: 输出目录
    """
    print("\n" + "="*60)
    print("生成可视化图表...")
    print("="*60)
    
    colors = {
        "GD": "#1f77b4",      # 蓝色
        "SGD": "#ff7f0e",     # 橙色
        "NAG": "#2ca02c",     # 绿色
        "Adam": "#d62728"     # 红色
    }
    
    linestyles = {
        "GD": "-",
        "SGD": "--",
        "NAG": "-.",
        "Adam": ":"
    }
    
    # 1. 训练损失对比图
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for result in results:
        iterations = range(1, len(result["train_loss"]) + 1)
        plt.plot(
            iterations,
            result["train_loss"],
            label=result["name"],
            color=colors[result["name"]],
            linestyle=linestyles[result["name"]],
            linewidth=2,
            alpha=0.8
        )
    plt.xlabel("Iterations", fontsize=16)
    plt.ylabel("Train loss", fontsize=16)
    plt.title("Train loss comparison", fontsize=18, fontweight="bold")
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=14)
    
    # 2. 测试损失对比图
    plt.subplot(2, 2, 2)
    for result in results:
        if result["test_loss"]:
            iterations = range(1, len(result["test_loss"]) + 1)
            plt.plot(
                iterations,
                result["test_loss"],
                label=result["name"],
                color=colors[result["name"]],
                linestyle=linestyles[result["name"]],
                linewidth=2,
                alpha=0.8
            )
    plt.xlabel("Iterations", fontsize=16)
    plt.ylabel("Test loss", fontsize=16)
    plt.title("Test loss comparison", fontsize=18, fontweight="bold")
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=14)
    
    # 3. 测试准确率对比图
    plt.subplot(2, 2, 3)
    for result in results:
        if result["test_accuracy"]:
            iterations = range(1, len(result["test_accuracy"]) + 1)
            plt.plot(
                iterations,
                result["test_accuracy"],
                label=result["name"],
                color=colors[result["name"]],
                linestyle=linestyles[result["name"]],
                linewidth=2,
                alpha=0.8
            )
    plt.xlabel("Iterations", fontsize=16)
    plt.ylabel("Test accuracy", fontsize=16)
    plt.title("Test accuracy comparison", fontsize=18, fontweight="bold")
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=14)
    
    # 4. 梯度范数对比图（对数尺度）
    plt.subplot(2, 2, 4)
    for result in results:
        if result["grad_norm"]:
            iterations = range(1, len(result["grad_norm"]) + 1)
            plt.plot(
                iterations,
                result["grad_norm"],
                label=result["name"],
                color=colors[result["name"]],
                linestyle=linestyles[result["name"]],
                linewidth=2,
                alpha=0.8
            )
    plt.xlabel("Iterations", fontsize=16)
    plt.ylabel("Gradient norm (log scale)", fontsize=16)
    plt.title("Gradient norm comparison", fontsize=18, fontweight="bold")
    plt.yscale("log")
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3, which="both")
    plt.tick_params(labelsize=14)
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "optimizer_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
    print(f"已保存对比图: {comparison_path}")
    plt.close()
    
    # 5. 性能指标汇总表（柱状图）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    names = [r["name"] for r in results]
    
    # 最终训练损失
    final_train_losses = [r["train_loss"][-1] for r in results]
    bars0 = axes[0, 0].bar(names, final_train_losses, color=[colors[n] for n in names], alpha=0.7)
    axes[0, 0].set_ylabel("Final train loss", fontsize=16)
    axes[0, 0].set_title("Final train loss comparison", fontsize=18, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3, axis="y")
    axes[0, 0].tick_params(labelsize=14)
    # 调整y轴范围，不从0开始
    if len(final_train_losses) > 0:
        min_val = min(final_train_losses)
        max_val = max(final_train_losses)
        range_val = max_val - min_val
        if range_val > 0:
            axes[0, 0].set_ylim(max(0, min_val - range_val * 0.15), max_val + range_val * 0.15)
    for i, v in enumerate(final_train_losses):
        axes[0, 0].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=13)
    
    # 最终测试损失
    final_test_losses = [r["test_loss"][-1] if r["test_loss"] else 0 for r in results]
    valid_test_losses = [v for v in final_test_losses if v > 0]
    bars1 = axes[0, 1].bar(names, final_test_losses, color=[colors[n] for n in names], alpha=0.7)
    axes[0, 1].set_ylabel("Final test loss", fontsize=16)
    axes[0, 1].set_title("Final test loss comparison", fontsize=18, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3, axis="y")
    axes[0, 1].tick_params(labelsize=14)
    # 调整y轴范围，不从0开始
    if len(valid_test_losses) > 0:
        min_val = min(valid_test_losses)
        max_val = max(valid_test_losses)
        range_val = max_val - min_val
        if range_val > 0:
            axes[0, 1].set_ylim(max(0, min_val - range_val * 0.15), max_val + range_val * 0.15)
    for i, v in enumerate(final_test_losses):
        if v > 0:
            axes[0, 1].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=13)
    
    # 最终测试准确率
    final_test_accs = [r["test_accuracy"][-1] if r["test_accuracy"] else 0 for r in results]
    valid_test_accs = [v for v in final_test_accs if v > 0]
    bars2 = axes[1, 0].bar(names, final_test_accs, color=[colors[n] for n in names], alpha=0.7)
    axes[1, 0].set_ylabel("Final test accuracy", fontsize=16)
    axes[1, 0].set_title("Final test accuracy comparison", fontsize=18, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3, axis="y")
    axes[1, 0].tick_params(labelsize=14)
    # 调整y轴范围，不从0开始
    if len(valid_test_accs) > 0:
        min_val = min(valid_test_accs)
        max_val = max(valid_test_accs)
        range_val = max_val - min_val
        if range_val > 0:
            axes[1, 0].set_ylim(max(0, min_val - range_val * 0.15), max_val + range_val * 0.15)
    for i, v in enumerate(final_test_accs):
        if v > 0:
            axes[1, 0].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=13)
    
    # 运行时间
    times = [r["time"] for r in results]
    bars3 = axes[1, 1].bar(names, times, color=[colors[n] for n in names], alpha=0.7)
    axes[1, 1].set_ylabel("Running time (seconds)", fontsize=16)
    axes[1, 1].set_title("Running time comparison", fontsize=18, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    axes[1, 1].tick_params(labelsize=14)
    # 调整y轴范围，不从0开始
    if len(times) > 0:
        min_val = min(times)
        max_val = max(times)
        range_val = max_val - min_val
        if range_val > 0:
            axes[1, 1].set_ylim(max(0, min_val - range_val * 0.15), max_val + range_val * 0.15)
    for i, v in enumerate(times):
        axes[1, 1].text(i, v, f"{v:.2f}s", ha="center", va="bottom", fontsize=13)
    
    plt.tight_layout()
    summary_path = os.path.join(output_dir, "optimizer_summary.png")
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    print(f"已保存汇总图: {summary_path}")
    plt.close()


def print_summary(results: List[Dict[str, any]]) -> None:
    """
    打印性能对比摘要。
    
    参数:
        results: 所有优化器的结果列表
    """
    print("\n" + "="*80)
    print("性能对比摘要")
    print("="*80)
    
    print(f"\n{'优化器':<10} {'最终训练损失':<15} {'最终测试损失':<15} "
          f"{'最终测试准确率':<15} {'运行时间(秒)':<15} {'迭代次数':<10}")
    print("-" * 80)
    
    for result in results:
        final_train_loss = result["train_loss"][-1]
        final_test_loss = result["test_loss"][-1] if result["test_loss"] else None
        final_test_acc = result["test_accuracy"][-1] if result["test_accuracy"] else None
        elapsed_time = result["time"]
        iterations = result["iterations"]
        
        test_loss_str = f"{final_test_loss:.6f}" if final_test_loss is not None else "N/A"
        test_acc_str = f"{final_test_acc:.4f}" if final_test_acc is not None else "N/A"
        
        print(f"{result['name']:<10} {final_train_loss:<15.6f} {test_loss_str:<15} "
              f"{test_acc_str:<15} {elapsed_time:<15.2f} {iterations:<10}")
    
    print("\n" + "="*80)


def main():
    """
    主函数：运行所有优化器并进行比较。
    """
    print("="*80)
    print("优化器性能比较: GD vs SGD vs NAG vs Adam")
    print("="*80)
    
    # 加载数据
    try:
        X_train, y_train, X_test, y_test = load_unified_data()
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("\n尝试使用 GD.py 的数据加载函数...")
        try:
            X_train, y_train = load_a9a_data(file_path="a9a.dat")
            X_test, y_test = load_a9a_data(file_path="a9at.dat")
            if X_train is None or X_test is None:
                print("数据加载失败，请确保数据文件存在。")
                return
        except Exception as e2:
            print(f"使用 GD.py 的数据加载函数也失败: {e2}")
            return
    
    # 统一超参数设置
    LAMBDA_REG = 0.01          # 正则化系数
    N_ITERATIONS = 200          # 迭代次数（统一）
    
    # 各优化器的特定超参数
    GD_LEARNING_RATE = 1.0
    SGD_LEARNING_RATE = 0.1
    SGD_BATCH_SIZE = 256
    SGD_EPOCHS = N_ITERATIONS
    NAG_LEARNING_RATE = 0.5
    NAG_MOMENTUM_BETA = 0.9
    ADAM_LEARNING_RATE = 0.01
    ADAM_BETA1 = 0.9
    ADAM_BETA2 = 0.999
    ADAM_EPS = 1e-8
    
    # 运行所有优化器
    results = []
    
    # GD
    try:
        gd_result = run_gd_optimizer(
            X_train, y_train, X_test, y_test,
            LAMBDA_REG, GD_LEARNING_RATE, N_ITERATIONS
        )
        results.append(gd_result)
    except Exception as e:
        print(f"GD 优化器运行失败: {e}")
    
    # SGD
    try:
        sgd_result = run_sgd_optimizer(
            X_train, y_train, X_test, y_test,
            LAMBDA_REG, SGD_LEARNING_RATE, SGD_EPOCHS, SGD_BATCH_SIZE
        )
        results.append(sgd_result)
    except Exception as e:
        print(f"SGD 优化器运行失败: {e}")
    
    # NAG
    try:
        nag_result = run_nag_optimizer(
            X_train, y_train, X_test, y_test,
            LAMBDA_REG, NAG_LEARNING_RATE, N_ITERATIONS, NAG_MOMENTUM_BETA
        )
        results.append(nag_result)
    except Exception as e:
        print(f"NAG 优化器运行失败: {e}")
    
    # Adam
    try:
        adam_result = run_adam_optimizer(
            X_train, y_train, X_test, y_test,
            LAMBDA_REG, ADAM_LEARNING_RATE, N_ITERATIONS,
            ADAM_BETA1, ADAM_BETA2, ADAM_EPS
        )
        results.append(adam_result)
    except Exception as e:
        print(f"Adam 优化器运行失败: {e}")
    
    if not results:
        print("错误: 所有优化器运行失败。")
        return
    
    # 打印摘要
    print_summary(results)
    
    # 可视化
    visualize_comparison(results)
    
    print("\n" + "="*80)
    print("比较完成！")
    print("="*80)


if __name__ == "__main__":
    main()

