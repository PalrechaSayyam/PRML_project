"""
task_hardness_tox21.py
----------------------
Computes Task Hardness metrics (INT_CHEM, EXT_CHEM, Task_Hardness)
for the Tox21 dataset using DeepChem + ChemBERTa embeddings + s-OTDD.

Steps:
1. Load Tox21 dataset (for labels & PR-AUC training).
2. Compute INT_CHEM = 1 - PR-AUC (Random Forest on ECFP).
3. Load precomputed ChemBERTa embeddings for each task from:
      chemical_embeddings/<task>_embeddings.npy
4. Compute EXT_CHEM using OTDD (Optimal Transport Dataset Distance)
   between labeled embeddings of each task.
5. Combine metrics into Task_Hardness = α * EXT_CHEM + β * INT_CHEM
"""

import os
import numpy as np
import pandas as pd
import deepchem as dc
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from otdd.pytorch.sotdd import compute_pairwise_distance
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier

# ---------------- Configuration ----------------
EMB_DIR = "chemical_embeddings"
SAVE_PATH = "results/task_hardness_tox21.csv"

N_JOBS = 4
SEED = 42
K_NEAREST = 5
ALPHA, BETA = 0.5, 0.5
MAX_SAMPLES = 500        # Limit samples per task for speed
NUM_PROJECTIONS = 2048   # Efficient number for fast OTDD computation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------------------------------


# ---------------- INT_CHEM Computation ----------------
# def compute_int_chem(X_train, y_train, X_valid, y_valid):
#     """Compute INT_CHEM = 1 - PR-AUC using RandomForest."""
#     if len(np.unique(y_train)) < 2:
#         return np.nan, np.nan
#     model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=SEED, n_jobs=-1)
#     model.fit(X_train, y_train)
#     y_prob = model.predict_proba(X_valid)[:, 1]
#     pr_auc = average_precision_score(y_valid, y_prob)
#     return 1 - pr_auc, pr_auc

def compute_int_chem_fewshot(X,y, m_train_list=[1024, 2048, 4096], seed=42):
    pr_aucs = []

    for m_train in m_train_list:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=m_train, random_state=seed)
        for train_idx, valid_idx in sss.split(X, y):
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            if len(np.unique(y_train)) < 2 or len(np.unique(y_valid)) < 2:
                print(f"skipping split for Task: {task} (split contains single class)")
                continue

            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
            
            # Train and evaluate
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_valid)[:, 1]
            pr_auc = average_precision_score(y_valid, y_prob)
            pr_aucs.append(pr_auc)
            
    return 1 - np.mean(pr_aucs), np.mean(pr_aucs)


# ---------------- Load Tox21 dataset ----------------
print("Loading Tox21 dataset...")
tasks, datasets, transformers = dc.molnet.load_tox21(featurizer="ECFP")
train_dataset, valid_dataset, test_dataset = datasets
print(f"Loaded {len(tasks)} tasks.\n")

# ---------------- Compute INT_CHEM per task ----------------
results = []
print("Computing INT_CHEM for each task...")

with ThreadPoolExecutor(max_workers=N_JOBS) as executor:
    futures = {}
    for i, task in enumerate(tasks):
        mask_train = ~np.isnan(train_dataset.y[:, i])
        mask_valid = ~np.isnan(valid_dataset.y[:, i])

        X_train = train_dataset.X[mask_train]
        y_train = train_dataset.y[mask_train, i].astype(int)
        X_valid = valid_dataset.X[mask_valid]
        y_valid = valid_dataset.y[mask_valid, i].astype(int)

        futures[executor.submit(compute_int_chem_fewshot, X_train, y_train)] = task

    for future in tqdm(as_completed(futures), total=len(futures)):
        task = futures[future]
        try:
            int_chem, pr_auc_val = future.result()
            results.append({"Task": task, "INT_CHEM": int_chem, "PR_AUC_val": pr_auc_val})
        except Exception as e:
            print(f"Error on task {task}: {e}")

df = pd.DataFrame(results).dropna().reset_index(drop=True)


# ---------------- EXT_CHEM via OTDD ----------------
print("\nComputing EXT_CHEM using OTDD distances between task embeddings...")

# Load embeddings per task
embeddings = {}
for task in tasks:
    path = os.path.join(EMB_DIR, f"{task}_embeddings.npy")
    if os.path.exists(path):
        arr = np.load(path)
        # if arr.shape[0] > MAX_SAMPLES:  # Subsample for speed
        #     idx = np.random.choice(arr.shape[0], MAX_SAMPLES, replace=False)
        #     arr = arr[idx]
        embeddings[task] = arr
    else:
        print(f"Embeddings not found for {task}, skipping EXT_CHEM for this task.")

# Compute pairwise OTDD distances
ext_chem = {}
task_list = list(embeddings.keys())
num_tasks = len(task_list)
dist_matrix = np.zeros((num_tasks, num_tasks))

from torch.utils.data import TensorDataset, DataLoader
# Params tuned for reasonable speed on CPU; raise num_projections if you have GPU/time
NUM_PROJECTIONS = 2000        # try 512-2048 for faster runs, increase for more precision
BATCH_SIZE = 128
KWARGS_OTDD = {
    "dimension": None,        # will set per-task from embedding dim
    "num_moments": 6,
    "use_conv": False,
    "precision": "float",     # use "double" if you want higher precision (slower)
    "p": 2,
    "chunk": 500
}

# Build dataloaders list in the same format used by the repo
dataloaders = []
task_list = []
for i, task in enumerate(tasks):
    mask = ~np.isnan(train_dataset.y[:, i])
    X = torch.tensor(embeddings[task], dtype=torch.float32)
    y = torch.tensor(train_dataset.y[mask, i].astype(int), dtype=torch.long)

    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    dataloaders.append(dl)
    task_list.append(task)

num_tasks = len(dataloaders)
print(f"Prepared {num_tasks} dataloaders for OTDD.")

# set dimension kwarg from embedding dimensionality
KWARGS_OTDD["dimension"] = embeddings[task_list[0]].shape[1]

# call compute_pairwise_distance exactly as in repo example
print("Computing s-OTDD pairwise distances (this may take time)...")
try:
    list_pairwise_dist = compute_pairwise_distance(list_D=dataloaders,device=DEVICE,num_projections=NUM_PROJECTIONS,evaluate_time=True, **KWARGS_OTDD)
except TypeError:
    # Some versions may return only the pairwise distances
    list_pairwise_dist = compute_pairwise_distance(
        list_D=dataloaders,
        device=DEVICE,
        num_projections=NUM_PROJECTIONS,
        evaluate_time=False,
        **KWARGS_OTDD
    )

# convert returned pairwise vector (len = n*(n-1)/2) into symmetric matrix
list_pairwise_dist = torch.tensor(list_pairwise_dist).cpu().numpy().ravel()
dist_matrix = np.zeros((num_tasks, num_tasks), dtype=float)
t = 0
for i in range(num_tasks):
    for j in range(i + 1, num_tasks):
        dist_matrix[i, j] = dist_matrix[j, i] = float(list_pairwise_dist[t])
        t += 1

# Compute EXT_CHEM = weighted avg of k nearest distances
ext_chem_values = {}
for i, target_task in enumerate(task_list):
    distances = dist_matrix[i]
    valid_mask = ~np.isnan(distances)
    valid_dists = distances[valid_mask]
    valid_tasks = np.array(task_list)[valid_mask]

    if len(valid_dists) > K_NEAREST:
        k_idx = np.argsort(valid_dists)[:K_NEAREST]
        nearest_dists = valid_dists[k_idx]
        nearest_tasks = valid_tasks[k_idx]
        weights = np.array([
            df.loc[df["Task"] == t, "INT_CHEM"].values[0] for t in nearest_tasks
        ])
        ext_val = np.average(nearest_dists, weights=weights)
    else:
        ext_val = np.nanmean(valid_dists)

    ext_chem_values[target_task] = ext_val

df["EXT_CHEM"] = df["Task"].map(ext_chem_values)
df["EXT_CHEM"] = (df["EXT_CHEM"] - df["EXT_CHEM"].min())/(df["EXT_CHEM"].max() - df["EXT_CHEM"].min())

# ---------------- Combine into overall hardness ----------------
df["Task_Hardness"] = ALPHA * df["EXT_CHEM"] + BETA * df["INT_CHEM"]

# ---------------- Save results ----------------
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
df = df[["Task", "PR_AUC_val", "INT_CHEM", "EXT_CHEM", "Task_Hardness"]]
df.to_csv(SAVE_PATH, index=False)

print(f"\nTask hardness metrics saved to: {SAVE_PATH}")
print(df)
