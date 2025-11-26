"""
task_trainer.py
Handles training & evaluation for all Tox21 tasks.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)
from concurrent.futures import ThreadPoolExecutor, as_completed


def train_single_task(task_tuple):
    """Train and evaluate RF & SVM for a single Tox21 task."""
    i, task, train_dataset, test_dataset = task_tuple

    # Extract non-NaN data
    y_train = train_dataset.y[:, i]
    mask_train = ~np.isnan(y_train)
    X_train = train_dataset.X[mask_train]
    y_train = y_train[mask_train]

    y_test = test_dataset.y[:, i]
    mask_test = ~np.isnan(y_test)
    X_test = test_dataset.X[mask_test]
    y_test = y_test[mask_test]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        print(f"Skipping {task} (insufficient data).")
        return []

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # ---- Random Forest ----
    rf = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=42
    )
    start = time.time()
    rf.fit(X_train, y_train)
    rf_time = time.time() - start

    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]

    rf_metrics = {
        "Task": task,
        "Model": "Random Forest",
        "Accuracy": accuracy_score(y_test, rf_pred),
        "Precision": precision_score(y_test, rf_pred, zero_division=0),
        "Recall": recall_score(y_test, rf_pred, zero_division=0),
        "F1": f1_score(y_test, rf_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, rf_prob),
        "PR-AUC": average_precision_score(y_test, rf_prob),
        "Train Time (s)": rf_time
    }
    results.append(rf_metrics)

    # ---- SVM ----
    svm = SVC(
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        random_state=42
    )
    start = time.time()
    svm.fit(X_train_scaled, y_train)
    svm_time = time.time() - start

    svm_pred = svm.predict(X_test_scaled)
    svm_prob = svm.predict_proba(X_test_scaled)[:, 1]

    svm_metrics = {
        "Task": task,
        "Model": "SVM",
        "Accuracy": accuracy_score(y_test, svm_pred),
        "Precision": precision_score(y_test, svm_pred, zero_division=0),
        "Recall": recall_score(y_test, svm_pred, zero_division=0),
        "F1": f1_score(y_test, svm_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, svm_prob),
        "PR-AUC": average_precision_score(y_test, svm_prob),
        "Train Time (s)": svm_time
    }
    results.append(svm_metrics)

    return results


def run_all_tasks(tasks, train_dataset, test_dataset, max_workers=3):
    """Run all Tox21 tasks in parallel (CPU-based)."""
    task_inputs = [(i, t, train_dataset, test_dataset) for i, t in enumerate(tasks)]

    all_results = []

    print(f"\nLaunching parallel training across {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(train_single_task, t) for t in task_inputs]
        for f in as_completed(futures):
            try:
                res = f.result()
                if res:
                    all_results.extend(res)
            except Exception as e:
                print(f"Error during training: {e}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/tox21_baseline_results.csv", index=False)

    print("\nResults saved to results/tox21_baseline_results.csv")

    # ---- Visualization ----
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x="Task", y="ROC-AUC", hue="Model")
    plt.xticks(rotation=45, ha='right')
    plt.title("ROC-AUC Comparison Across Tasks")
    plt.tight_layout()
    plt.savefig("results/plots/roc_auc.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x="Task", y="PR-AUC", hue="Model")
    plt.xticks(rotation=45, ha='right')
    plt.title("PR-AUC Comparison Across Tasks")
    plt.tight_layout()
    plt.savefig("results/plots/pr_auc.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=results_df, x="Task", y="Train Time (s)", hue="Model")
    plt.xticks(rotation=45, ha='right')
    plt.title("Training Time per Task")
    plt.tight_layout()
    plt.savefig("results/plots/train_time.png")
    plt.close()

    print("\nAll plots saved in results/plots/")
    return results_df
