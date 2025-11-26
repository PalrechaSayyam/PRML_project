## PRML_project
# My coursework term project: "A Task-Hardness Driven Hybrid Modeling Framework for Molecular Toxicity Prediction"

# Hybrid Modeling Framework for Molecular Toxicity Prediction (Tox21)

This project develops a **task-aware hybrid machine learning framework** for molecular toxicity prediction using the **Tox21** dataset. The work evaluates classical ML models, graph neural networks (GNNs), task-hardness descriptors, and a new hybrid model that integrates complementary molecular representations.

---

## Overview

Predicting chemical toxicity is crucial in early drug discovery. However, datasets like Tox21 pose challenges such as **class imbalance**, **noisy assays**, and **varying task difficulty**. This project benchmarks multiple modeling approaches and develops a hybrid model that addresses these issues.

---

## Models Evaluated

### Classical ML Models
- Random Forest (RF)
- Support Vector Machine (SVM)
- XGBoost  
With improvements using:
- SMOTE oversampling  
- Optuna hyperparameter tuning  

### Graph Neural Networks
- Graph Convolutional Network (GraphConv)  
- Graph Attention Network (GAT)

### Task Hardness Analysis
Two descriptors were computed to understand model behavior across tasks:
- **INT-CHEM** — intrinsic task difficulty  
- **EXT-CHEM** — inter-task similarity (via Optimal Transport)

This analysis reveals which tasks favor classical ML vs. GNNs.

---

## Hybrid Model (Proposed)

A new hybrid model combining:
- **GNN-derived embeddings**
- **ECFP fingerprints**

A Random Forest classifier is trained on the concatenated features:

X_hybrid = [GNN_embeddings || ECFP_fingerprint]

This design leverages both learned graph structure and handcrafted chemical motifs.

---

## Key Result

The hybrid model achieves the **best overall performance** across all Tox21 tasks:

- **Highest average PR-AUC and ROC-AUC**
- **100–300% PR-AUC improvement** over baseline RF on many tasks

It demonstrates strong complementarity between classical and learned molecular representations.

---

## Summary

This project provides:
- A benchmark of ML and GNN models on Tox21  
- Insights from task-hardness descriptors  
- A hybrid modeling strategy that significantly improves toxicity prediction  

