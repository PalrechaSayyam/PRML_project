"""
gnn_tox21_GAT.py
GAT (Graph Attention Network) implementation for Tox21 using DeepChem + PyTorch (CPU-only)
"""

import os
import numpy as np
import deepchem as dc
from deepchem.models import GATModel
from deepchem.metrics import Metric, roc_auc_score, prc_auc_score
from sklearn.metrics import average_precision_score
from deepchem.feat import MolGraphConvFeaturizer
import torch
import pandas as pd

# ---------------- Configuration ----------------
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def train_gnn_tox21(seed=42, epochs=30, batch_size=64, save_csv=True):
    """
    Trains a GAT model on Tox21 dataset using CPU.
    Returns: metrics dictionary: { 'VALIDATION': {...}, 'TEST': {...} }
    """

    print("Loading Tox21 dataset...")
    featurizer = MolGraphConvFeaturizer(use_edges=True)

    tasks, datasets, transformers = dc.molnet.load_tox21(
        featurizer=featurizer
    )
    train_dataset, valid_dataset, test_dataset = datasets

    print("Dataset loaded.")
    print(f"Tasks: {tasks}")

    # Force CPU mode (DeepChem uses torch internally)
    torch.device("cpu")

    # Define metrics
    # metric_roc = Metric(roc_auc_score, mode="classification")
    # metric_pr = Metric(prc_auc_score, mode="classification")

    print("\nInitializing GAT Model (Graph Attention Network)...")

    model = GATModel(
        n_tasks=len(tasks),
        mode="classification",
        n_attention_heads=4,
        graph_attention_layers=[64, 64],
        dropout=0.2,
        batch_size=batch_size,
        learning_rate=1e-3,
        random_state=seed,
        use_gpu=False,
    )

    print("GAT model initialized. Training on CPU...")

    model.fit(train_dataset, nb_epoch=epochs)

    print("\nEvaluating model...")

    results_rows = []

    for split_name, dataset in [("VALID", valid_dataset), ("TEST", test_dataset)]:
        preds = model.predict(dataset)              # shape: (samples, tasks, 2 classes)
        y_true = dataset.y                          # true labels

        for task_idx, task in enumerate(tasks):
            true = y_true[:, task_idx]
            pred = preds[:, task_idx, 1]            # probability of class=1

            mask = ~np.isnan(true)

            if np.sum(mask) > 0:
                roc_val = roc_auc_score(true[mask], pred[mask])
                pr_val = average_precision_score(true[mask], pred[mask]) 
            else:
                roc_val, pr_val = np.nan, np.nan

            results_rows.append([
                "GraphConv", task, split_name,
                float(roc_val), float(pr_val)
            ])
            
    df = pd.DataFrame(results_rows,
                      columns=["Model", "Task", "Split", "ROC-AUC", "PR-AUC"])
    
    if save_csv:
        df.to_csv(os.path.join(RESULTS_DIR,"tox21_gnn_results_GAT.csv"), index=False)

    # print("\nFinal Results (GAT):")
    # print(results)

    return df
