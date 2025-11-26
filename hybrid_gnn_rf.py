"""
Hybrid RF Experiment on Tox21 (Corrected)
---------------------------------------
This script computes a hybrid representation:
    X_hybrid = concat([GNN_embeddings, ECFP_fingerprints])

1.  Trains a GNN on the train set.
2.  Uses model.predict_embedding() to get *true* embeddings.
3.  Concatenates with ECFP.
4.  Trains a task-specific RandomForest on these features.
"""

import deepchem as dc
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from deepchem.metrics import Metric

# ------------------------------------------------------------
# 1) Train GNN model (once)
# ------------------------------------------------------------
def train_gnn_model(tasks, train_graph, valid_graph):
    """
    Trains a GNN on the train set, using the valid set
    for early stopping (via ValidationCallback).
    Returns: A *trained* model object.
    """
    print("Training GraphConv model...")
    model = dc.models.GraphConvModel(
        n_tasks=len(tasks),
        mode="classification",
        batch_size=64,
        learning_rate=1e-3,
        dropout=0.2,
        # We need model_dir so the callback can save the best model
        model_dir="./results/models/gnn_model",
        use_gpu=False,
    )
    
    # --- THIS IS THE FIX ---
    # 1. Define the metric we want to monitor: PR-AUC
    #    We set 'mode="classification"' to ensure it calculates correctly.
    metric = Metric(average_precision_score, mode="classification", classification_handling_mode = "direct")
    
    # 2. Use ValidationCallback, not EarlyStoppingCallback
    # This will check the valid_graph every `interval` steps
    # An interval of 100 steps is ~1 epoch (6264/64 = 98 steps/epoch)
    callback = dc.models.callbacks.ValidationCallback(
        valid_graph, interval=100, metrics=[metric]
    )
    # --- END OF FIX ---

    print("Training GNN (monitoring PR-AUC for early stopping)...")
    model.fit(train_graph, nb_epoch=100, callbacks=[callback])

    # Restore the best performing model based on the callback
    print("Restoring best model from validation...")
    model.restore()
    
    return model

# ------------------------------------------------------------
# 2) Load ECFP fingerprints
# ------------------------------------------------------------
def get_ecfp_fps(dataset):
    """
    dataset: ECFP DeepChem dataset
    returns: numpy array (N, 1024)
    """
    try:
        return dataset.X
    except:
        return np.array([x for x in dataset.X])


# ------------------------------------------------------------
# 3) Train a single RF model per task (fixed hyperparameters)
# ------------------------------------------------------------
def train_and_eval_rf(tasks, X_train, y_train, w_train, X_valid, y_valid, w_valid):
    """
    Trains a separate RF for each task on the hybrid features.
    Handles masking internally.
    """
    results = []
    print("\nTraining RandomForest on hybrid features...")

    for t_idx, task in enumerate(tasks):
        # Get labels and weights for the current task
        # We must handle the masked (missing) labels
        mask_train = w_train[:, t_idx].astype(bool)
        mask_valid = w_valid[:, t_idx].astype(bool)
        
        y_train_task = y_train[mask_train, t_idx].astype(int)
        X_train_task = X_train[mask_train]
        
        y_valid_task = y_valid[mask_valid, t_idx].astype(int)
        X_valid_task = X_valid[mask_valid]

        if len(np.unique(y_train_task)) < 2:
            print(f"[WARN] Task {task} has single class in train. Skipping.")
            continue

        # FIXED RF hyperparameters
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train_task, y_train_task)
        
        # Guard against single-class validation set
        if len(np.unique(y_valid_task)) < 2:
            print(f"[WARN] Task {task} has single class in valid. Skipping.")
            continue
            
        y_prob = model.predict_proba(X_valid_task)[:, 1]

        pr = average_precision_score(y_valid_task, y_prob)
        roc = roc_auc_score(y_valid_task, y_prob)

        results.append({
            "Task": task,
            "PR-AUC": pr,
            "ROC-AUC": roc,
        })

        print(f"{task:15s} | PR-AUC = {pr:.4f}   ROC-AUC = {roc:.4f}")

    df = pd.DataFrame(results)
    return df


# ------------------------------------------------------------
# 4) Main function
# ------------------------------------------------------------
def run_hybrid_experiment():
    print("Loading Tox21 dataset with GraphConv featurizer (MASTER)...")
    tasks, graph_sets, _ = dc.molnet.load_tox21(featurizer="GraphConv")
    train_graph, valid_graph, test_graph = graph_sets

    print(f"Master train dataset size: {len(train_graph)}")
    print(f"Master valid dataset size: {len(valid_graph)}")

    # --- Step 1: Train GNN ---
    gnn_model = train_gnn_model(tasks, train_graph, valid_graph)

    # --- Step 2: Get *true* GNN embeddings ---
    print("Extracting GNN embeddings...")
    X_train_gnn = gnn_model.predict_embedding(train_graph)
    X_valid_gnn = gnn_model.predict_embedding(valid_graph)
    # X_train_gnn shape is (6272, N)

    # --- Step 3: Get ECFP features *for the same molecules* ---
    print("Generating ECFP features for the *same* molecules...")
    ecfp_featurizer = dc.feat.CircularFingerprint(size=1024, radius=2)
    
    # We must iterate to find featurization failures
    X_train_ecfp_list = []
    X_valid_ecfp_list = []
    train_indices_to_keep = []
    valid_indices_to_keep = []
    
    # Process training set
    print("Robustly featurizing ECFP for train set...")
    for i, smiles in enumerate(train_graph.ids):
        fp = ecfp_featurizer.featurize([smiles]) # Featurize one at a time
        if fp.shape[0] > 0: # Check if featurization was successful (fp[0] is not None)
            X_train_ecfp_list.append(fp[0])
            train_indices_to_keep.append(i)
        # else: print(f"ECFP failed for train SMILES: {smiles}") # Uncomment to see fails
            
    # Process validation set
    print("Robustly featurizing ECFP for valid set...")
    for i, smiles in enumerate(valid_graph.ids):
        fp = ecfp_featurizer.featurize([smiles])
        if fp.shape[0] > 0:
            X_valid_ecfp_list.append(fp[0])
            valid_indices_to_keep.append(i)

    # Convert lists to numpy arrays
    X_train_ecfp = np.array(X_train_ecfp_list)
    X_valid_ecfp = np.array(X_valid_ecfp_list)
    # X_train_ecfp shape is now (6264, 1024)

    # --- Step 4: Filter *all* datasets to the common, successful indices ---
    print("Filtering all datasets to common successful molecules...")
    
    # Filter GNN embeddings to match
    X_train_gnn = X_train_gnn[train_indices_to_keep]
    X_valid_gnn = X_valid_gnn[valid_indices_to_keep]
    # X_train_gnn shape is now (6264, N)

    # Filter labels and weights (y, w) to match
    y_train = train_graph.y[train_indices_to_keep]
    w_train = train_graph.w[train_indices_to_keep]
    
    y_valid = valid_graph.y[valid_indices_to_keep]
    w_valid = valid_graph.w[valid_indices_to_keep]

    print(f"Filtered train GNN shape: {X_train_gnn.shape}")
    print(f"Filtered train ECFP shape: {X_train_ecfp.shape}")
    print(f"Filtered valid GNN shape: {X_valid_gnn.shape}")
    print(f"Filtered valid ECFP shape: {X_valid_ecfp.shape}")

    # --- Step 5: Build Hybrid Feature Set ---
    # Now, the dimensions *must* match
    print("Concatenating features...")
    X_train_hybrid = np.concatenate([X_train_gnn, X_train_ecfp], axis=1)
    X_valid_hybrid = np.concatenate([X_valid_gnn, X_valid_ecfp], axis=1)

    print(f"Hybrid feature shape: {X_train_hybrid.shape}")

    # --- Step 6: Train and Evaluate RF ---
    df_results = train_and_eval_rf(
        tasks,
        X_train_hybrid, y_train, w_train,
        X_valid_hybrid, y_valid, w_valid
    )

    df_results.to_csv("results/hybrid_gnn_rf_results.csv", index=False)
    print("\nSaved results -> results/hybrid_gnn_rf_results.csv")

    return df_results

# ------------------------------------------------------------
# If run as script
# ------------------------------------------------------------
if __name__ == "__main__":
    df = run_hybrid_experiment()
    print("\nFinal Hybrid Results (Corrected):")
    print(df)