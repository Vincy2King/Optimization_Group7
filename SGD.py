import os
import requests
from typing import Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file


# ===============================
# 1. 下载并加载 a9a 数据集
# ===============================

def get_a9a_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    下载并加载 a9a 数据集（LIBSVM 格式），标签为 {-1, +1}，返回稠密矩阵。
    """
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    url_train = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a"
    url_test = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t"

    train_path = os.path.join(data_dir, "a9a")
    test_path = os.path.join(data_dir, "a9a.t")

    if not os.path.exists(train_path):
        print("Downloading a9a train set...")
        r = requests.get(url_train, timeout=30)
        r.raise_for_status()
        with open(train_path, "wb") as f:
            f.write(r.content)
        print("Train set downloaded.")

    if not os.path.exists(test_path):
        print("Downloading a9a test set...")
        r = requests.get(url_test, timeout=30)
        r.raise_for_status()
        with open(test_path, "wb") as f:
            f.write(r.content)
        print("Test set downloaded.")

    n_features = 123  # 标准维度
    X_train, y_train = load_svmlight_file(train_path, n_features=n_features)
    X_test, y_test = load_svmlight_file(test_path, n_features=n_features)

    # 标签保持为 {-1, +1}
    return X_train.toarray(), y_train, X_test.toarray(), y_test


# ===============================
# 2. 数学函数：sigmoid / loss / grad
# ===============================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定版 Sigmoid."""
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def compute_loss(X: np.ndarray, t: np.ndarray, w: np.ndarray, lambda_reg: float) -> float:
    """
    F(w) = (1/n) sum log(1 + exp(-t_i x_i^T w)) + (lambda/2)||w||^2
    """
    n_samples, n_features = X.shape
    assert t.shape[0] == n_samples
    assert w.shape[0] == n_features

    z = t * (X @ w)
    loss_logistic = np.mean(np.log1p(np.exp(-z)))  # log(1 + exp(-z))
    reg_term = 0.5 * lambda_reg * np.dot(w, w)
    return loss_logistic + reg_term


def compute_gradient_batch(Xb: np.ndarray, tb: np.ndarray, w: np.ndarray,
                           lambda_reg: float) -> np.ndarray:
    """
    mini-batch 梯度:
    grad = (1/B) Xb^T[-t * sigmoid(-t x^T w)] + lambda w
    """
    z = tb * (Xb @ w)
    grad_logistic = Xb.T @ (-tb * sigmoid(-z)) / Xb.shape[0]
    grad_reg = lambda_reg * w
    return grad_logistic + grad_reg


# ===============================
# 3. Mini-batch SGD 主体
# ===============================

def sgd_logreg(
    X: np.ndarray,
    t: np.ndarray,
    lambda_reg: float = 0.01,
    learning_rate: float = 0.1,
    n_epochs: int = 400,
    batch_size: int = 256,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    random_state: int = 42,
):
    """
    Mini-batch SGD for L2-regularized logistic regression (labels in {-1, +1}).

    返回:
        w,
        train_loss_history,
        test_loss_history,
        test_acc_history,
        grad_norm_history
    """
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X.shape
    w = np.zeros(n_features)

    train_loss_hist: List[float] = []
    test_loss_hist: List[float] = []
    test_acc_hist: List[float] = []
    grad_norm_hist: List[float] = []

    print(f"Start SGD: lr={learning_rate}, batch_size={batch_size}, "
          f"lambda={lambda_reg}, epochs={n_epochs}")

    for epoch in range(n_epochs):
        indices = rng.permutation(n_samples)

        # 一次 epoch: 遍历所有 mini-batch
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            Xb = X[batch_idx]
            tb = t[batch_idx]

            grad = compute_gradient_batch(Xb, tb, w, lambda_reg)
            w -= learning_rate * grad

        # 记录当前 epoch 的指标（用全训练集评估一次）
        train_loss = compute_loss(X, t, w, lambda_reg)
        train_loss_hist.append(train_loss)
        grad_norm_hist.append(float(np.linalg.norm(grad)))

        if X_test is not None and y_test is not None:
            tl = compute_loss(X_test, y_test, w, lambda_reg)
            test_loss_hist.append(tl)
            y_pred = np.sign(X_test @ w)
            acc = float(np.mean(y_pred == y_test))
            test_acc_hist.append(acc)

        if (epoch + 1) % 5 == 0:
            if X_test is not None and y_test is not None:
                print(
                    f"Epoch {epoch+1}/{n_epochs} | "
                    f"Train Loss={train_loss:.6f} | "
                    f"Test Loss={test_loss_hist[-1]:.6f} | "
                    f"Test Acc={test_acc_hist[-1]:.4f} | "
                    f"Gradient={grad_norm_hist[-1]:.4f}"
                )
            else:
                print(f"Epoch {epoch+1}/{n_epochs} | Train Loss={train_loss:.6f}")

    print("SGD training finished.")
    return w, train_loss_hist, test_loss_hist, test_acc_hist, grad_norm_hist


# ===============================
# 4. 主程序：只看 SGD + 收敛图
# ===============================

if __name__ == "__main__":
    # 1) 加载数据
    X_train, y_train, X_test, y_test = get_a9a_data()
    print(f"Loaded data. Train: {X_train.shape}, Test: {X_test.shape}")

    # 2) 添加偏置项
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    print(f"After bias. Train: {X_train.shape}, Test: {X_test.shape}")

    # 3) 训练 SGD
    LAMBDA = 0.01
    LR = 0.1
    EPOCHS = 400
    BATCH = 512

    w_sgd, loss_sgd, test_loss_sgd, test_acc_sgd, grad_norm_sgd = sgd_logreg(
        X=X_train,
        t=y_train,
        lambda_reg=LAMBDA,
        learning_rate=LR,
        n_epochs=EPOCHS,
        batch_size=BATCH,
        X_test=X_test,
        y_test=y_test,
    )

    # 4) 画收敛曲线（风格和对方那张类似：三联图）
    epochs = range(1, EPOCHS + 1)
    plt.figure(figsize=(15, 5))

    # 图1：训练 & 测试 Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, loss_sgd, label="Train Loss", linewidth=2)
    if test_loss_sgd:
        plt.plot(epochs, test_loss_sgd, label="Test Loss", linewidth=2, linestyle="--")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("SGD: Loss Convergence")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 图2：测试集 Accuracy
    plt.subplot(1, 3, 2)
    if test_acc_sgd:
        plt.plot(epochs, test_acc_sgd, label="Test Accuracy", linewidth=2, color="green")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.title("SGD: Test Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.axis("off")

    # 图3：梯度范数（log 轴）
    plt.subplot(1, 3, 3)
    plt.plot(epochs, grad_norm_sgd, label="Gradient Norm", linewidth=2, color="red")
    plt.yscale("log")
    plt.xlabel("Iterations")
    plt.ylabel("Gradient Norm")
    plt.title("SGD: Gradient Norm (Log Scale)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sgd_convergence_a9a.png", dpi=300, bbox_inches="tight")
    print("Saved figure: sgd_convergence_a9a.png")
    plt.show()

    # 5) 最终测试集表现
    final_test_loss = compute_loss(X_test, y_test, w_sgd, LAMBDA)
    final_test_pred = np.sign(X_test @ w_sgd)
    final_test_acc = float(np.mean(final_test_pred == y_test))
    print(f"\n[SGD] Final Test Loss = {final_test_loss:.6f}, Test Acc = {final_test_acc:.4f}")
