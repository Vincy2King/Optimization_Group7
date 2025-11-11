#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adam optimizer on a9a (logistic loss + L2), generating three plots:
- ADAM_loss_curve.png
- ADAM_accuracy_curve.png
- ADAM_gradient_norm_curve.png

Designed to mirror the visualization style used in NAG.py so you can compare methods.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.datasets import load_svmlight_file

# ---------- Data Loading (dense, with bias column like NAG.py) ----------
def load_a9a_dense(data_dir: str = "data") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_path = os.path.join(data_dir, "a9a")
    test_path  = os.path.join(data_dir, "a9a.t")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError("Expect data files at ./data/a9a and ./data/a9a.t (LIBSVM).")
    Xtr, ytr = load_svmlight_file(train_path)
    Xte, yte = load_svmlight_file(test_path, n_features=Xtr.shape[1])
    Xtr = Xtr.toarray(); Xte = Xte.toarray()
    ytr = ytr.astype(np.float64); yte = yte.astype(np.float64)
    # add bias column (intercept) to match NAG.py style
    Xtr = np.hstack([np.ones((Xtr.shape[0], 1)), Xtr])
    Xte = np.hstack([np.ones((Xte.shape[0], 1)), Xte])
    return Xtr, ytr, Xte, yte

# ---------- Math helpers (match NAG.py formulas) ----------
def sigmoid(z: np.ndarray) -> np.ndarray:
    # stable sigmoid
    return np.where(z >= 0, 1.0/(1.0+np.exp(-z)), np.exp(z)/(1.0+np.exp(z)))

def logistic_loss_with_l2(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> float:
    s = y * (X @ w)
    # stable: log(1 + exp(-s)) = logaddexp(0, -s)
    loss = np.mean(np.logaddexp(0.0, -s))
    reg = 0.5 * lam * np.dot(w, w)
    return loss + reg

def grad_logistic_l2(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    s = y * (X @ w)
    # -y * sigmoid(-s) = -y / (1 + exp(s))
    g_factor = -y * (1.0/(1.0+np.exp(s)))
    grad = (X.T @ g_factor) / X.shape[0] + lam * w
    return grad

# ---------- Adam Optimizer (full-batch, mirrors NAG loop length) ----------
class Adam:
    def __init__(self, dim: int, beta1=0.9, beta2=0.999, eps=1e-8):
        self.m = np.zeros(dim, dtype=np.float64)
        self.v = np.zeros(dim, dtype=np.float64)
        self.beta1 = beta1; self.beta2 = beta2; self.eps = eps
        self.t = 0

    def step(self, w: np.ndarray, g: np.ndarray, lr: float) -> np.ndarray:
        self.t += 1
        self.m = self.beta1 * self.m + (1-self.beta1) * g
        self.v = self.beta2 * self.v + (1-self.beta2) * (g*g)
        mhat = self.m / (1 - self.beta1**self.t)
        vhat = self.v / (1 - self.beta2**self.t)
        w = w - lr * mhat / (np.sqrt(vhat) + self.eps)
        return w

def train_adam(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray,
    lam: float = 1e-2, lr: float = 1e-2, iters: int = 200,
    beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8
):
    n, d = Xtr.shape
    w = np.zeros(d, dtype=np.float64)
    opt = Adam(d, beta1, beta2, eps)
    loss_hist, test_loss_hist, test_acc_hist, grad_norm_hist = [], [], [], []

    for k in range(1, iters+1):
        g = grad_logistic_l2(Xtr, ytr, w, lam)
        w = opt.step(w, g, lr=lr)

        # records
        loss_hist.append(logistic_loss_with_l2(Xtr, ytr, w, lam))
        test_loss_hist.append(logistic_loss_with_l2(Xte, yte, w, lam))
        s_te = Xte @ w
        y_pred = np.where(s_te >= 0.0, 1.0, -1.0)
        test_acc_hist.append(np.mean(y_pred == yte))
        grad_norm_hist.append(np.linalg.norm(g))

        if k % 20 == 0 or k == 1 or k == iters:
            print(f"[iter {k:3d}] train_loss={loss_hist[-1]:.5f} | "
                  f"test_loss={test_loss_hist[-1]:.5f} | test_acc={test_acc_hist[-1]*100:.2f}% | "
                  f"||g||={grad_norm_hist[-1]:.4e}")

    return w, loss_hist, test_loss_hist, test_acc_hist, grad_norm_hist

# ---------- Main & Plots (same style/structure as NAG.py; filenames prefixed with ADAM_) ----------
if __name__ == "__main__":
    Xtr, ytr, Xte, yte = load_a9a_dense(".")
    print(f"Loaded: train {Xtr.shape}, test {Xte.shape}")

    LAMBDA_REG = 1e-2     # 与 NAG.py 的 0.01 一致
    LEARNING_RATE = 1e-2  # Adam 通常更稳，lr 可取 1e-2
    N_ITER = 200          # 与 NAG.py 保持相同迭代数便于对比
    B1, B2, EPS = 0.9, 0.999, 1e-8

    w, loss_hist, test_loss_hist, test_acc_hist, grad_norm_hist = train_adam(
        Xtr, ytr, Xte, yte,
        lam=LAMBDA_REG, lr=LEARNING_RATE, iters=N_ITER, beta1=B1, beta2=B2, eps=EPS
    )

    # 1) Loss curve (train + test)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, N_ITER+1), loss_hist, label='Training Loss (Adam)', linewidth=2)
    plt.plot(range(1, N_ITER+1), test_loss_hist, label='Test Loss', linewidth=2, linestyle='--')
    plt.xlabel('Iterations'); plt.ylabel('Loss'); plt.title('Loss Convergence Curve (Adam)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('ADAM_loss_curve.png', dpi=300, bbox_inches='tight'); plt.close()
    print("Saved: ADAM_loss_curve.png")

    # 2) Test accuracy curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, N_ITER+1), test_acc_hist, label='Test Accuracy', linewidth=2, color='green')
    plt.xlabel('Iterations'); plt.ylabel('Accuracy'); plt.title('Test Accuracy Curve (Adam)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('ADAM_accuracy_curve.png', dpi=300, bbox_inches='tight'); plt.close()
    print("Saved: ADAM_accuracy_curve.png")

    # 3) Gradient norm curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, N_ITER+1), grad_norm_hist, label='Gradient Norm', linewidth=2, color='red')
    plt.xlabel('Iterations'); plt.ylabel('||grad||'); plt.title('Gradient Norm (Adam)')
    plt.yscale('log'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('ADAM_gradient_norm_curve.png', dpi=300, bbox_inches='tight'); plt.close()
    print("Saved: ADAM_gradient_norm_curve.png")

    # final metrics
    final_test_loss = logistic_loss_with_l2(Xte, yte, w, LAMBDA_REG)
    y_pred = np.where(Xte @ w >= 0.0, 1.0, -1.0)
    final_acc = np.mean(y_pred == yte)
    print(f"[final] test_loss={final_test_loss:.6f} | test_acc={final_acc*100:.2f}%")
