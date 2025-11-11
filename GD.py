import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
import os
import time

# --- 1. Data Loading and Preprocessing ---

def load_a9a_data(file_path='a9a'):
    """
    Load the a9a dataset and perform preprocessing.

    - Convert the sparse matrix to a dense NumPy array.
    - Append a bias (intercept) term to the feature matrix.

    Returns:
        X (np.array): Feature matrix with bias term (n_samples, n_features + 1)
        y (np.array): Label vector (n_samples,)
    """
    if not os.path.exists(file_path):
        alt_paths = [f"{file_path}.dat", f"{file_path}.txt"]
        for alt in alt_paths:
            if os.path.exists(alt):
                print(f"Info: Detected dataset file '{alt}', using it instead.")
                file_path = alt
                break

    try:
        # Use scikit-learn to load the LIBSVM formatted data.
        # n_features=123 is the standard dimensionality for a9a.
        X_sparse, y = load_svmlight_file(file_path, n_features=123)
    except FileNotFoundError:
        print(f"Error: Dataset file '{file_path}' not found.")
        print("Please download the a9a dataset from the LIBSVM repository and place it alongside this script.")
        return None, None

    # Convert the sparse matrix to a dense NumPy array
    X = X_sparse.toarray()
    
    # Retrieve number of samples and original features
    n_samples, n_features = X.shape
    print(f"Loaded dataset: {n_samples} samples, {n_features} features.")

    # Append a bias term (intercept) by adding a column of ones
    X = np.hstack((X, np.ones((n_samples, 1))))
    
    print(f"Bias term appended. Feature dimension is now: {X.shape[1]}")
    
    # Ensure labels are +1 and -1 (load_svmlight_file typically handles this)
    y[y == 0] = -1
    
    return X, y

# --- 2. Core Math Utilities ---

def calculate_cost(X, y, w, lambda_val):
    """
    Compute the objective function F(w).
    F(w) = (1/n) * Σ log(1 + exp(-y_i * x_i^T w)) + (λ/2) * ||w||_2^2
    """
    n_samples = X.shape[0]
    
    # Compute x_i^T * w
    scores = X @ w
    
    # Compute loss term using np.logaddexp for numerical stability
    loss = np.sum(np.logaddexp(0, -y * scores)) / n_samples
    
    # Compute regularization term
    reg_term = (lambda_val / 2) * np.linalg.norm(w)**2
    
    return loss + reg_term

def calculate_gradient(X, y, w, lambda_val):
    """
    Compute the gradient of F(w).
    ∇F(w) = (1/n) * Σ [σ(-y_i * x_i^T w) * (-y_i * x_i)] + λ * w
    where σ(z) = 1 / (1 + exp(-z))
    """
    n_samples = X.shape[0]
    
    # Compute x_i^T * w
    scores = X @ w
    
    # Compute sigmoid σ(-y_i * x_i^T w) = 1 / (1 + exp(y_i * x_i^T w))
    sigmoid_vals = 1 / (1 + np.exp(y * scores))
    
    # Vectorized gradient of the loss term
    loss_gradient = X.T @ (sigmoid_vals * -y) / n_samples
    
    # Gradient of the regularization term
    reg_gradient = lambda_val * w
    
    return loss_gradient + reg_gradient

# --- 3. Batch Gradient Descent Implementation ---

def gradient_descent(
    X,
    y,
    lambda_val,
    learning_rate,
    max_iters,
    X_val=None,
    y_val=None,
):
    """
    Run batch gradient descent and optionally evaluate on a validation set.
    """
    n_features = X.shape[1]
    w = np.zeros(n_features)

    cost_history = []
    val_loss_history = []
    val_accuracy_history = []
    grad_norm_history = []

    print("\n--- Starting Gradient Descent ---")
    
    for i in range(max_iters):
        # 1. Compute current cost
        cost = calculate_cost(X, y, w, lambda_val)
        cost_history.append(cost)
        
        # Progress log
        if i % 50 == 0 or i == max_iters - 1:
            print(f"Iteration {i:4d}/{max_iters}: F(w) = {cost:.6f}")
            
        # 2. Compute gradient
        grad = calculate_gradient(X, y, w, lambda_val)
        
        # 3. Update weights
        w = w - learning_rate * grad
        grad_norm_history.append(np.linalg.norm(grad))

        if X_val is not None and y_val is not None:
            val_loss = calculate_cost(X_val, y_val, w, lambda_val)
            val_loss_history.append(val_loss)

            val_scores = X_val @ w
            val_pred = np.where(val_scores >= 0, 1, -1)
            val_accuracy = np.mean(val_pred == y_val)
            val_accuracy_history.append(val_accuracy)
 
    print("--- Gradient Descent Completed ---\n")
    return w, cost_history, val_loss_history, val_accuracy_history, grad_norm_history

# --- 4. Hyperparameter Search with scikit-learn ---

def run_logistic_regression_grid_search(X, y, results_dir=None):
    """
    Use GridSearchCV to tune logistic regression hyperparameters such as C (=1/λ).
    """
    print("\n--- Running GridSearchCV for Hyperparameter Tuning ---")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "solver": ["liblinear", "saga"],
        "fit_intercept": [True, False]
    }

    log_reg = LogisticRegression(
        max_iter=4000,
        penalty="l2",
        tol=1e-4,
        n_jobs=None,
        verbose=0
    )

    grid = GridSearchCV(
        estimator=log_reg,
        param_grid=param_grid,
        scoring="neg_log_loss",
        cv=5,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    grid.fit(X_train, y_train)

    print(f"Best parameter combination: {grid.best_params_}")
    print(f"Best cross-validation negative log loss: {grid.best_score_:.6f}")

    best_model = grid.best_estimator_
    val_proba = best_model.predict_proba(X_val)
    val_loss = log_loss(y_val, val_proba)

    print(f"Validation log loss: {val_loss:.6f}")
    print("--- GridSearchCV Completed ---\n")

    results_df = pd.DataFrame(grid.cv_results_)
    results_df["mean_log_loss"] = -results_df["mean_test_score"]
    results_df["std_log_loss"] = results_df["std_test_score"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for solver, group in results_df.groupby("param_solver"):
        ax.errorbar(
            group["param_C"].astype(float),
            group["mean_log_loss"],
            yerr=group["std_log_loss"],
            marker="o",
            capsize=3,
            label=f"solver={solver}"
        )

    ax.set_xscale("log")
    ax.set_xlabel("C (log scale)", fontsize=12)
    ax.set_ylabel("Cross-validated log loss", fontsize=12)
    ax.set_title("GridSearchCV Performance by Parameter", fontsize=16)
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()

    if results_dir is not None:
        plot_path = os.path.join(results_dir, "grid_search_log_loss.png")
        plt.savefig(plot_path, dpi=300)
        print(f"GridSearchCV plot saved to: {plot_path}")

    plt.show()

    return grid, val_loss

        # (假设之前的代码已经运行完毕，我们已经有了 final_w)
def load_and_preprocess_test_data(file_path='a9at.dat'):
    """加载并预处理测试数据，确保其格式与训练数据一致。"""
    try:
        X_sparse, y = load_svmlight_file(file_path, n_features=123)
        X = X_sparse.toarray()
        n_samples, _ = X.shape
        # 关键：必须进行和训练数据完全相同的预处理，即添加偏置项
        X = np.hstack((X, np.ones((n_samples, 1))))
        y[y == 0] = -1
        print(f"\n成功加载测试数据集: {n_samples} 个样本。")
        return X, y
    except FileNotFoundError:
        print(f"错误：测试集文件 '{file_path}' 未找到。")
        return None, None

def predict(X, w):
    """使用训练好的权重 w 进行预测。"""
    scores = X @ w
    # 预测结果为 +1 或 -1
    predictions = np.sign(scores)
    # np.sign(0) 会返回 0，我们将其处理为 +1 (或 -1，这是一个约定)
    predictions[predictions == 0] = 1
    return predictions

def calculate_accuracy(y_true, y_pred):
    """计算预测准确率。"""
    accuracy = np.mean(y_true == y_pred)
    return accuracy

# --- 5. Main Execution and Result Saving ---

if __name__ == "__main__":
    # --- Hyperparameters ---
    LAMBDA = 0.001          # L2 regularization strength
    LEARNING_RATE = 1     # Learning rate
    MAX_ITERS = 400        # Maximum iterations
    RUN_GRID_SEARCH = False  # Whether to run GridSearchCV
    
    # --- Result directory ---
    RESULTS_DIR = "gd_results"
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created result directory: {RESULTS_DIR}")

    # 1. Load data
    X_train, y_train = load_a9a_data(file_path='a9a.dat')
    X_val, y_val = load_a9a_data(file_path='a9at.dat')
 
    if X_train is not None:
        # 2. Run gradient descent
        start_time = time.time()
        final_w, cost_history, val_loss_history, val_accuracy_history, grad_norm_history = gradient_descent(
            X=X_train, 
            y=y_train, 
            lambda_val=LAMBDA, 
            learning_rate=LEARNING_RATE, 
            max_iters=MAX_ITERS,
            X_val=X_val,
            y_val=y_val
        )
        end_time = time.time()
        
        # 3. Summary
        print("--- Optimization Summary ---")
        print(f"Hyperparameters: λ = {LAMBDA}, η = {LEARNING_RATE}, iterations = {MAX_ITERS}")
        print(f"Elapsed time: {end_time - start_time:.2f} seconds")
        print(f"Initial cost: {cost_history[0]:.6f}")
        print(f"Final cost: {cost_history[-1]:.6f}")
        print(f"L2 norm of final weights: {np.linalg.norm(final_w):.4f}")
        if val_loss_history:
            print(f"Final validation cost: {val_loss_history[-1]:.6f}")
            print(f"Final validation accuracy: {val_accuracy_history[-1]:.4f}")
        elif X_val is None:
            print("Validation dataset not found; skipping validation metrics.")

        if RUN_GRID_SEARCH:
            run_logistic_regression_grid_search(X_train, y_train, results_dir=RESULTS_DIR)

        # 4. Save results
        w_path = os.path.join(RESULTS_DIR, "final_w.txt")
        cost_path = os.path.join(RESULTS_DIR, "cost_history.txt")
        
        np.savetxt(w_path, final_w, fmt="%.8f", header="Final weight vector w")
        np.savetxt(cost_path, cost_history, fmt="%.8f", header="Cost F(w) per iteration")
        print(f"\nFinal weight vector saved to: {w_path}")
        print(f"Cost history saved to: {cost_path}")

        # 5. Visualization (aligned with NAG style)
        iterations = range(1, MAX_ITERS + 1)
        has_validation = bool(val_loss_history)

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(15, 5))

        # Subplot 1: training and validation loss
        plt.subplot(1, 3, 1)
        plt.plot(iterations, cost_history, label='Training Loss', linewidth=2)
        if has_validation:
            plt.plot(iterations, val_loss_history, label='Validation Loss', linewidth=2, linestyle='--')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss Convergence Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 2: validation accuracy (if available)
        plt.subplot(1, 3, 2)
        if has_validation:
            plt.plot(iterations, val_accuracy_history, label='Validation Accuracy', linewidth=2, color='green')
            plt.ylim(0.75, 0.865)
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No validation data', ha='center', va='center', fontsize=12)
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy Curve')
        plt.grid(True, alpha=0.3)

        # Subplot 3: gradient norm
        plt.subplot(1, 3, 3)
        plt.plot(iterations, grad_norm_history, label='Gradient Norm', linewidth=2, color='red')
        plt.xlabel('Iteration')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norm Curve')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        overview_path = os.path.join(RESULTS_DIR, 'gd_convergence_overview.png')
        plt.savefig(overview_path, dpi=300, bbox_inches='tight')
        print(f"Saved GD overview plot to: {overview_path}")
        plt.show()

        # Individual plots (loss, accuracy, gradient norm)
        plt.figure(figsize=(8, 6))
        plt.plot(iterations, cost_history, label='Training Loss', linewidth=2)
        if has_validation:
            plt.plot(iterations, val_loss_history, label='Validation Loss', linewidth=2, linestyle='--')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss Convergence Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        loss_path = os.path.join(RESULTS_DIR, 'gd_loss_curve.png')
        plt.savefig(loss_path, dpi=300, bbox_inches='tight')
        print(f"Saved loss curve to: {loss_path}")
        plt.close()

        plt.figure(figsize=(8, 6))
        if has_validation:
            plt.plot(iterations, val_accuracy_history, label='Validation Accuracy', linewidth=2, color='green')
            plt.ylim(0.75, 0.865)
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No validation data', ha='center', va='center', fontsize=12)
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy Curve')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        acc_path = os.path.join(RESULTS_DIR, 'gd_accuracy_curve.png')
        plt.savefig(acc_path, dpi=300, bbox_inches='tight')
        print(f"Saved accuracy curve to: {acc_path}")
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(iterations, grad_norm_history, label='Gradient Norm', linewidth=2, color='red')
        plt.xlabel('Iteration')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norm Curve')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        grad_path = os.path.join(RESULTS_DIR, 'gd_gradient_norm_curve.png')
        plt.savefig(grad_path, dpi=300, bbox_inches='tight')
        print(f"Saved gradient norm curve to: {grad_path}")
        plt.close()

    if 'final_w' in locals() and final_w is not None:
        print("\n--- 开始模型评估 ---")
        X_test, y_test = load_and_preprocess_test_data(file_path='a9at.dat')
        
        if X_test is not None:
            # 使用训练得到的 final_w 进行预测
            y_pred = predict(X_test, final_w)
            
            # 计算准确率
            accuracy = calculate_accuracy(y_test, y_pred)
            
            print(f"模型在测试集上的准确率 (Accuracy): {accuracy:.4f} ({accuracy * 100:.2f}%)")
            
            # 将评估结果也保存下来
            accuracy_path = os.path.join(RESULTS_DIR, "test_accuracy.txt")
            with open(accuracy_path, 'w') as f:
                f.write(f"Test Set Accuracy: {accuracy:.4f}\n")
            print(f"测试集准确率已保存至: {accuracy_path}")