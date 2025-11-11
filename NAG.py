import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
import os
import requests
from typing import Tuple, List, Optional

# --- 1. 数据准备：自动下载并加载 a9a 数据集 ---
def get_a9a_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    下载、解压并使用 scikit-learn 加载 a9a 数据集。
    处理训练集和测试集特征维度不一致的问题。
    
    返回:
        X_train: 训练特征矩阵 (n_train_samples, n_features)
        y_train: 训练标签向量 {-1, 1} (n_train_samples,)
        X_test: 测试特征矩阵 (n_test_samples, n_features)
        y_test: 测试标签向量 {-1, 1} (n_test_samples,)
    
    异常:
        requests.RequestException: 当下载失败时抛出
        ValueError: 当数据加载失败时抛出
    """
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    url_train = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a"
    url_test = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t"
    
    train_path = os.path.join(data_dir, "a9a")
    test_path = os.path.join(data_dir, "a9a.t")

    # 下载训练集
    if not os.path.exists(train_path):
        print("正在下载 a9a 训练集...")
        try:
            r = requests.get(url_train, timeout=30)
            r.raise_for_status()
            with open(train_path, 'wb') as f:
                f.write(r.content)
            print("下载完成。")
        except requests.RequestException as e:
            raise requests.RequestException(f"下载训练集失败: {e}")

    # 下载测试集
    if not os.path.exists(test_path):
        print("正在下载 a9a 测试集...")
        try:
            r = requests.get(url_test, timeout=30)
            r.raise_for_status()
            with open(test_path, 'wb') as f:
                f.write(r.content)
            print("下载完成。")
        except requests.RequestException as e:
            raise requests.RequestException(f"下载测试集失败: {e}")
        
    # a9a 数据集的标准特征维度是 123。测试集可能缺少最后一列。
    # 我们通过 n_features=123 来确保加载时维度一致。
    n_features = 123
    try:
        X_train, y_train = load_svmlight_file(train_path, n_features=n_features)
        X_test, y_test = load_svmlight_file(test_path, n_features=n_features)
    except Exception as e:
        raise ValueError(f"加载数据失败: {e}")

    # load_svmlight_file 加载的标签是 +1 和 -1，这正是我们需要的 t_i
    # 将稀疏矩阵转为密集矩阵，方便计算
    return X_train.toarray(), y_train, X_test.toarray(), y_test

# --- 2. 模型和数学函数定义 ---
def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    稳定的 Sigmoid 函数，防止数值溢出。
    
    参数:
        z: 输入数组
    
    返回:
        Sigmoid 函数值，范围在 (0, 1)
    """
    # 防止 exp 溢出
    return np.piecewise(z, [z > 0], [
        lambda x: 1 / (1 + np.exp(-x)),
        lambda x: np.exp(x) / (1 + np.exp(x))
    ])

def compute_loss(X: np.ndarray, t: np.ndarray, w: np.ndarray, lambda_reg: float) -> float:
    """
    计算 L2 正则化逻辑回归的总损失。
    
    参数:
        X: 特征矩阵 (n_samples, n_features)
        t: 标签向量 {-1, 1} (n_samples,)
        w: 权重向量 (n_features,)
        lambda_reg: L2 正则化系数
    
    返回:
        总损失值（逻辑损失 + L2 正则化项）
    
    异常:
        ValueError: 当输入维度不匹配时抛出
    """
    if X.shape[0] != t.shape[0]:
        raise ValueError(f"样本数不匹配: X.shape[0]={X.shape[0]}, t.shape[0]={t.shape[0]}")
    if X.shape[1] != w.shape[0]:
        raise ValueError(f"特征维度不匹配: X.shape[1]={X.shape[1]}, w.shape[0]={w.shape[0]}")
    n_samples = X.shape[0]
    # Logistic Loss: log(1 + exp(-t_i * x_i^T * w))
    logistic_loss = np.sum(np.log(1 + np.exp(-t * (X @ w)))) / n_samples
    # L2 Regularization
    reg_term = (lambda_reg / 2) * np.linalg.norm(w)**2
    return logistic_loss + reg_term

def compute_gradient(X: np.ndarray, t: np.ndarray, w: np.ndarray, lambda_reg: float) -> np.ndarray:
    """
    计算 L2 正则化逻辑回归的梯度。
    
    参数:
        X: 特征矩阵 (n_samples, n_features)
        t: 标签向量 {-1, 1} (n_samples,)
        w: 权重向量 (n_features,)
        lambda_reg: L2 正则化系数
    
    返回:
        梯度向量 (n_features,)
    
    异常:
        ValueError: 当输入维度不匹配时抛出
    """
    if X.shape[0] != t.shape[0]:
        raise ValueError(f"样本数不匹配: X.shape[0]={X.shape[0]}, t.shape[0]={t.shape[0]}")
    if X.shape[1] != w.shape[0]:
        raise ValueError(f"特征维度不匹配: X.shape[1]={X.shape[1]}, w.shape[0]={w.shape[0]}")
    n_samples = X.shape[0]
    # 计算 h - y 或类似形式更稳定
    # grad_logistic = (1/n) * sum_{i} (sigmoid(t_i * x_i^T * w) - 1) * t_i * x_i
    # 一个更简洁的推导结果是：
    # grad = (1/n) * X.T @ (-t * sigmoid(-t * (X @ w)))
    
    z = t * (X @ w)
    # 梯度中的 logistic 部分
    grad_logistic = X.T @ (-t * sigmoid(-z)) / n_samples
    
    # 梯度中的正则化部分
    grad_reg = lambda_reg * w
    
    return grad_logistic + grad_reg

# --- 3. Nesterov 加速梯度 (NAG) 优化器 ---
def nesterov_accelerated_gradient(
    X: np.ndarray,
    t: np.ndarray,
    lambda_reg: float,
    learning_rate: float,
    n_iterations: int,
    momentum_beta: float,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, List[float], List[float], List[float], List[float]]:
    """
    Nesterov Accelerated Gradient (NAG) 算法实现。

    参数:
        X: 训练特征矩阵 (n_samples, n_features)
        t: 训练标签向量 {-1, 1} (n_samples,)
        lambda_reg: L2 正则化系数
        learning_rate: 学习率 (eta)
        n_iterations: 迭代次数
        momentum_beta: 动量参数 (beta)，通常在 [0, 1) 范围内
        X_test: 测试特征矩阵，可选 (n_test_samples, n_features)
        y_test: 测试标签向量，可选 (n_test_samples,)

    返回:
        w: 最终的权重向量 (n_features,)
        loss_history: 每次迭代的训练集损失值记录
        test_loss_history: 每次迭代的测试集损失值记录（如果提供了测试集）
        test_accuracy_history: 每次迭代的测试集准确率记录（如果提供了测试集）
        grad_norm_history: 每次迭代的梯度范数记录
    
    异常:
        ValueError: 当输入参数无效时抛出
    """
    if learning_rate <= 0:
        raise ValueError(f"学习率必须大于0，当前值: {learning_rate}")
    if not (0 <= momentum_beta < 1):
        raise ValueError(f"动量参数必须在 [0, 1) 范围内，当前值: {momentum_beta}")
    if n_iterations <= 0:
        raise ValueError(f"迭代次数必须大于0，当前值: {n_iterations}")
    if lambda_reg < 0:
        raise ValueError(f"正则化系数必须非负，当前值: {lambda_reg}")
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    # 在 NAG 中，我们需要保存上一步的权重 w_{k-1}
    # 为了统一循环，我们用 v (动量项)来表示 w_k - w_{k-1}
    v = np.zeros(n_features)
    loss_history = []
    test_loss_history = []
    test_accuracy_history = []
    grad_norm_history = []

    print(f"开始 NAG 训练... (lr={learning_rate}, beta={momentum_beta}, lambda={lambda_reg})")
    for k in range(n_iterations):
        # 1. "Lookahead" step: 计算动量更新后的临时点 y_k
        # y_k = w_k + beta * (w_k - w_{k-1})
        # 使用动量项 v 更高效: v_k = beta * v_{k-1} - eta * grad(w_{k-1})
        # NAG的经典形式：
        w_lookahead = w + momentum_beta * v

        # 2. 在 "Lookahead" 点计算梯度
        grad = compute_gradient(X, t, w_lookahead, lambda_reg)

        # 3. 更新动量项 v
        v = momentum_beta * v - learning_rate * grad

        # 4. 更新权重 w
        w = w + v

        # 记录训练集损失函数值
        train_loss = compute_loss(X, t, w, lambda_reg)
        loss_history.append(train_loss)

        # 记录梯度范数
        grad_norm = np.linalg.norm(grad)
        grad_norm_history.append(grad_norm)

        # 如果提供了测试集，记录测试集损失和准确率
        if X_test is not None and y_test is not None:
            test_loss = compute_loss(X_test, y_test, w, lambda_reg)
            test_loss_history.append(test_loss)

            # 计算测试集准确率
            y_pred_test = np.sign(X_test @ w)
            accuracy = np.mean(y_pred_test == y_test)
            test_accuracy_history.append(accuracy)

        if (k + 1) % 10 == 0:
            print(f"迭代 {k+1}/{n_iterations}, 训练损失: {train_loss:.6f}", end="")
            if X_test is not None and y_test is not None:
                print(f", 测试损失: {test_loss_history[-1]:.6f}, 测试准确率: {test_accuracy_history[-1]:.4f}")
            else:
                print()

    print("NAG 训练完成。")
    return w, loss_history, test_loss_history, test_accuracy_history, grad_norm_history


# --- 4. 主执行逻辑 ---
if __name__ == "__main__":
    # 加载数据
    X_train, y_train, X_test, y_test = get_a9a_data()
    print(f"数据加载完毕。训练集: {X_train.shape}, 测试集: {X_test.shape}")

    # 添加偏置项 (intercept term)
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    print(f"添加偏置项后。训练集: {X_train.shape}, 测试集: {X_test.shape}")

    # 设置超参数
    LAMBDA_REG = 0.01          # 正则化强度
    LEARNING_RATE = 0.5        # 学习率 (NAG 通常可以用比 GD 大一点的学习率)
    N_ITERATIONS = 400         # 迭代次数
    MOMENTUM_BETA = 0.9        # 动量参数
    
    # 运行 NAG 算法
    w_final_nag, loss_history_nag, test_loss_history_nag, test_accuracy_history_nag, grad_norm_history_nag = nesterov_accelerated_gradient(
        X=X_train,
        t=y_train,
        lambda_reg=LAMBDA_REG,
        learning_rate=LEARNING_RATE,
        n_iterations=N_ITERATIONS,
        momentum_beta=MOMENTUM_BETA,
        X_test=X_test,
        y_test=y_test
    )

    # --- 5. 结果可视化 ---
    # 绘制损失曲线（整体图）
    plt.figure(figsize=(15, 5))
    
    # 子图1: 训练和测试损失
    plt.subplot(1, 3, 1)
    plt.plot(range(1, N_ITERATIONS + 1), loss_history_nag, label=f'Training Loss (β={MOMENTUM_BETA})', linewidth=2)
    if test_loss_history_nag:
        plt.plot(range(1, N_ITERATIONS + 1), test_loss_history_nag, label='Test Loss', linewidth=2, linestyle='--')
    plt.xlabel('Iterations)')
    plt.ylabel('Loss')
    plt.title('Loss Convergence Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 测试准确率
    if test_accuracy_history_nag:
        plt.subplot(1, 3, 2)
        plt.plot(range(1, N_ITERATIONS + 1), test_accuracy_history_nag, label='Test Accuracy', linewidth=2, color='green')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title('Test Accuracy Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 子图3: 梯度范数
    plt.subplot(1, 3, 3)
    plt.plot(range(1, N_ITERATIONS + 1), grad_norm_history_nag, label='Gradient Norm', linewidth=2, color='red')
    plt.xlabel('Iterations)')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm Curve')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('NAG_convergence_curve.png', dpi=300, bbox_inches='tight')
    print("已保存整体图: NAG_convergence_curve.png")
    plt.show()
    
    # 单独保存每个子图
    # 子图1: 训练和测试损失
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, N_ITERATIONS + 1), loss_history_nag, label=f'Training Loss (β={MOMENTUM_BETA})', linewidth=2)
    if test_loss_history_nag:
        plt.plot(range(1, N_ITERATIONS + 1), test_loss_history_nag, label='Test Loss', linewidth=2, linestyle='--')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss Convergence Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('NAG_loss_curve.png', dpi=300, bbox_inches='tight')
    print("已保存损失曲线图: NAG_loss_curve.png")
    plt.close()
    
    # 子图2: 测试准确率
    if test_accuracy_history_nag:
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, N_ITERATIONS + 1), test_accuracy_history_nag, label='Test Accuracy', linewidth=2, color='green')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title('Test Accuracy Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('NAG_accuracy_curve.png', dpi=300, bbox_inches='tight')
        print("已保存准确率曲线图: NAG_accuracy_curve.png")
        plt.close()
    
    # 子图3: 梯度范数
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, N_ITERATIONS + 1), grad_norm_history_nag, label='梯度范数', linewidth=2, color='red')
    plt.xlabel('Iterations')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm Curve')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('NAG_gradient_norm_curve.png', dpi=300, bbox_inches='tight')
    print("已保存梯度范数曲线图: NAG_gradient_norm_curve.png")
    plt.close()

    # 在测试集上评估最终模型的性能
    final_test_loss = compute_loss(X_test, y_test, w_final_nag, LAMBDA_REG)
    print(f"\nLoss in Test Set: {final_test_loss:.6f}")

    # 计算准确率
    y_pred_test = np.sign(X_test @ w_final_nag)
    accuracy = np.mean(y_pred_test == y_test)
    print(f"Accuracy in Test Set: {accuracy:.4f}")
