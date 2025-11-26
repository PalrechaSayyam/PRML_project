"""
optimized_pipeline.py

Efficient version:
- Runs 3-4 representative Tox21 tasks in parallel.
- Uses DeepChem's built-in validation dataset.
- RF (baseline) + Optuna-tuned RF & XGBoost (8 trials each).
- SHAP applied on both baseline and best XGBoost model.
- Parallelization at task level.
"""

import os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import deepchem as dc
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
import optuna
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------- Configuration ----------------
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_DIR = os.path.join(RESULTS_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

SEED = 42
OPTUNA_TRIALS = 8
MAX_WORKERS = 12
USE_SMOTE = True
# TASKS_TO_RUN = ["NR-AR", "NR-ER", "SR-HSE", "SR-p53"]
# ------------------------------------------------


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "PR-AUC": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    }


# --- Optuna objective functions ---
def optuna_rf(trial, X_train, y_train, X_valid, y_valid):
    params = {
        "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 400]),
        "max_depth": trial.suggest_categorical("max_depth", [10, 20, None]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"])
    }
    clf = RandomForestClassifier(random_state=SEED, n_jobs=1, **params)
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_valid)[:, 1]
    return roc_auc_score(y_valid, prob)


def optuna_xgb(trial, X_train, y_train, X_valid, y_valid):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
        "subsample": trial.suggest_uniform("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.6, 1.0),
    }
    clf = XGBClassifier(use_label_encoder=False,eval_metric="logloss",random_state=SEED,**params)
    clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    prob = clf.predict_proba(X_valid)[:, 1]
    return roc_auc_score(y_valid, prob)

RESULTS_FILE = "tox21_optimized_results_new.csv"

def save_partial_result(result_dict):
    """Appends a single model result as a new row in CSV."""
    df_row = pd.DataFrame([result_dict])

    if not os.path.exists(RESULTS_FILE):
        df_row.to_csv(os.path.join(RESULTS_DIR, RESULTS_FILE), index=False)
    else:
        df_row.to_csv(os.path.join(RESULTS_DIR, RESULTS_FILE), mode='a', header=False, index=False)
    
def run_task(task_index, task_name, train_dataset, valid_dataset, test_dataset):
    results = []
    print(f"Starting task: {task_name}")

    # Extract data for the particular task
    y_train = train_dataset.y[:, task_index]
    y_valid = valid_dataset.y[:, task_index]
    y_test = test_dataset.y[:, task_index]

    mask_train = ~np.isnan(y_train)
    mask_valid = ~np.isnan(y_valid)
    mask_test = ~np.isnan(y_test)

    X_train = train_dataset.X[mask_train]
    X_valid = valid_dataset.X[mask_valid]
    X_test = test_dataset.X[mask_test]

    y_train = y_train[mask_train].astype(int)
    y_valid = y_valid[mask_valid].astype(int)
    y_test = y_test[mask_test].astype(int)

    # ----- Baseline RF -----
    # rf_base = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=SEED, n_jobs=-1)
    # rf_base.fit(X_train, y_train)
    # y_prob_base = rf_base.predict_proba(X_test)[:, 1]
    # y_pred_base = rf_base.predict(X_test)

    # base_metrics = compute_metrics(y_test, y_pred_base, y_prob_base)
    # base_metrics.update({"Task": task_name, "Model": "RF_Baseline"})
    # results.append(base_metrics)
    # save_partial_result(base_metrics)

    # SMOTE oversampling
    if USE_SMOTE:
        X_train, y_train = SMOTE(random_state=SEED).fit_resample(X_train, y_train)

    # --- Optuna RF ---
    study_rf = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study_rf.optimize(lambda t: optuna_rf(t, X_train, y_train, X_valid, y_valid), n_trials=OPTUNA_TRIALS, n_jobs=1)

    rf_best = RandomForestClassifier(random_state=SEED, **study_rf.best_params)
    rf_best.fit(X_train, y_train)
    y_prob_rf_opt = rf_best.predict_proba(X_test)[:, 1]
    y_pred_rf_opt = rf_best.predict(X_test)

    rf_metrics = compute_metrics(y_test, y_pred_rf_opt, y_prob_rf_opt)
    rf_metrics.update({"Task": task_name, "Model": "RF_Optimized"})
    results.append(rf_metrics)
    save_partial_result(rf_metrics)
    joblib.dump(rf_best, os.path.join(MODEL_DIR, f"{task_name}_RF_optimized.pkl"))

    # --- Optuna XGBoost ---
    study_xgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study_xgb.optimize(lambda t: optuna_xgb(t, X_train, y_train, X_valid, y_valid), n_trials=OPTUNA_TRIALS, n_jobs=1)

    xgb_best = XGBClassifier(use_label_encoder=False,eval_metric="logloss",random_state=SEED,**study_xgb.best_params)
    xgb_best.fit(X_train, y_train)
    y_prob_xgb_opt = xgb_best.predict_proba(X_test)[:, 1]
    y_pred_xgb_opt = xgb_best.predict(X_test)

    xgb_metrics = compute_metrics(y_test, y_pred_xgb_opt, y_prob_xgb_opt)
    xgb_metrics.update({"Task": task_name, "Model": "XGB_Optimized"})
    results.append(xgb_metrics)
    save_partial_result(xgb_metrics)
    joblib.dump(xgb_best, os.path.join(MODEL_DIR, f"{task_name}_XGB_optimized.pkl"))

    print(f"Saved models for task: {task_name}")

    return results


def main():
    print("\nLoading Tox21 dataset...")
    tasks, datasets, _ = dc.molnet.load_tox21(featurizer="ECFP")
    train_dataset, valid_dataset, test_dataset = datasets

    # selected_tasks = [t for t in TASKS_TO_RUN if t in tasks]
    selected_tasks = tasks
    print(f"Running tasks: {selected_tasks}")

    all_results = []
    task_inputs = [(tasks.index(t), t, train_dataset, valid_dataset, test_dataset)
                   for t in selected_tasks]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_task, *args) for args in task_inputs]
        for f in as_completed(futures):
            res = f.result()
            if res:
                all_results.extend(res)

    df = pd.DataFrame(all_results)
    # df.to_csv(os.path.join(RESULTS_DIR, "tox21_optimized_results_new.csv"), index=False)

    # print("\nResults saved: results/tox21_optimized_results.csv")
    print("Models saved: results/models/")

    return df


if __name__ == "__main__":
    main()