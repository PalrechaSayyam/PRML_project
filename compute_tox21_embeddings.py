"""
compute_tox21_embeddings.py
------------------------------------
Generates molecule-level embeddings for each Tox21 task using
the pretrained ChemBERTa-77M-MLM model.

Output:
    ./chemical_embeddings/<task_name>_embeddings.npy

Requirements:
    pip install deepchem transformers torch tqdm
"""

import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import deepchem as dc
from tqdm import tqdm

# ---------------- Configuration ----------------
MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"
BATCH_SIZE = 32
SAVE_DIR = "chemical_embeddings"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)
# ------------------------------------------------


def compute_chemberta_embeddings(smiles_list, tokenizer, model, batch_size=32):
    """Compute ChemBERTa embeddings using mean pooling on last_hidden_state."""
    all_embeddings = []

    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Computing embeddings", leave=False):
        batch = smiles_list[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=128)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # mean pooling
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # L2 normalization

        all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


def main():
    print(f"Using device: {DEVICE}")
    print(f"Loading ChemBERTa model ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    print("Loading Tox21 dataset...")
    tasks, datasets, transformers = dc.molnet.load_tox21(featurizer="Raw")
    train_dataset, valid_dataset, test_dataset = datasets
    dataset = train_dataset  

    print(f"Found {len(tasks)} tasks.")

    for i, task in enumerate(tasks):
        print(f"\nProcessing task: {task}")

        mask = ~np.isnan(dataset.y[:, i])
        smiles_list = np.array(dataset.ids)[mask]

        if len(smiles_list) == 0:
            print(f"Skipping {task}: no valid molecules.")
            continue

        embeddings = compute_chemberta_embeddings(smiles_list.tolist(), tokenizer, model, BATCH_SIZE)

        np.save(os.path.join(SAVE_DIR, f"{task}_embeddings.npy"), embeddings)
        print(f"Saved {task}: {embeddings.shape}")

    print("\nAll embeddings saved in ./chemical_embeddings/")


if __name__ == "__main__":
    main()
