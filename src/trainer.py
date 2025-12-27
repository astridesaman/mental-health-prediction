from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
import torch.optim as optim

from src.dataset import MentalDataset


# =========================
# 1) Chargement & preprocessing
# =========================

def load_and_preprocess(path: str):
    """
    Charge train.csv, applique le prétraitement et renvoie :
    X_train, y_train, X_val, y_val, mean, std, feature_cols
    """
    df = pd.read_csv(path)

    target_col = "Depression"

    # Drop colonnes inutiles
    df = df.drop(columns=["id", "Name"])

    num_cols = [
        "Age",
        "Academic Pressure",
        "Work Pressure",
        "CGPA",
        "Study Satisfaction",
        "Job Satisfaction",
        "Work/Study Hours",
        "Financial Stress",
    ]

    cat_cols = [
        "Gender",
        "City",
        "Working Professional or Student",
        "Profession",
        "Sleep Duration",
        "Dietary Habits",
        "Degree",
        "Have you ever had suicidal thoughts ?",
        "Family History of Mental Illness",
    ]

    y = df[target_col].astype("float32").values
    X = df.drop(columns=[target_col])

    # NA
    X[num_cols] = X[num_cols].fillna(X[num_cols].median(numeric_only=True))
    X[cat_cols] = X[cat_cols].fillna("Unknown")

    # one-hot
    X = pd.get_dummies(X, columns=cat_cols)

    feature_cols = X.columns.tolist()

    X = X.astype("float32").values

    # split 80/20
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))

    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # standardisation
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    return X_train, y_train, X_val, y_val, mean, std, feature_cols


def preprocess_test(df_test: pd.DataFrame, feature_cols: List[str], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Applique le même preprocessing au test:
    - drop id, Name
    - NA
    - one-hot
    - alignement des colonnes avec feature_cols
    - standardisation avec mean/std du train
    """
    df = df_test.copy()

    for col in ["id", "Name"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    num_cols = [
        "Age",
        "Academic Pressure",
        "Work Pressure",
        "CGPA",
        "Study Satisfaction",
        "Job Satisfaction",
        "Work/Study Hours",
        "Financial Stress",
    ]

    cat_cols = [
        "Gender",
        "City",
        "Working Professional or Student",
        "Profession",
        "Sleep Duration",
        "Dietary Habits",
        "Degree",
        "Have you ever had suicidal thoughts ?",
        "Family History of Mental Illness",
    ]

    df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    df = pd.get_dummies(df, columns=cat_cols)

    # align colonnes
    df = df.reindex(columns=feature_cols, fill_value=0)

    X = df.astype("float32").values
    X = (X - mean) / (std + 1e-8)
    return X


@torch.inference_mode()
def predict_on_test(
    model: torch.nn.Module,
    test_csv_path: str,
    feature_cols: List[str],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retourne: ids, probas, preds
    """
    df_test = pd.read_csv(test_csv_path)
    if "id" not in df_test.columns:
        raise ValueError("test.csv doit contenir une colonne 'id'.")

    ids = df_test["id"].to_numpy()

    X_test = preprocess_test(df_test, feature_cols, mean, std)
    X_tensor = torch.from_numpy(np.asarray(X_test, dtype="float32")).to(device)

    model.to(device)
    model.eval()

    probas = model(X_tensor).detach().cpu().numpy().reshape(-1)
    preds = (probas >= 0.5).astype(int)

    return ids, probas, preds


def generate_csv_probabilities(output_path: str, ids: np.ndarray, probabilities: np.ndarray) -> None:
    df = pd.DataFrame({
        "id": ids,
        "Depression_Probability": probabilities
    })
    df.to_csv(output_path, index=False)


def generate_kaggle_submission(output_path: str, ids: np.ndarray, preds: np.ndarray) -> None:
    df = pd.DataFrame({
        "id": ids,
        "Depression": preds
    })
    df.to_csv(output_path, index=False)


# =========================
# 2) Trainer
# =========================

@dataclass
class MentalHealthTrainer:
    batch_size: int
    n_epochs: int
    eval_samples: int

    def train(
        self,
        model: torch.nn.Module,
        train_set: MentalDataset,
        val_set: MentalDataset,
        optimizer: optim.Optimizer,
        device: torch.device,
        model_name: str = "model",
    ) -> float:
        """
        Entraîne UN modèle et renvoie la meilleure accuracy validation.
        Sauvegarde les meilleurs poids dans best_model_weights_{model_name}.pt
        """
        model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n=== Training {model_name} ({n_params:,} params) ===")

        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

        best_val = 0.0
        patience = 3
        counter = 0

        for epoch in range(self.n_epochs):
            model.train()
            train_loss = 0.0
            train_acc = 0.0
            n_train = 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                preds = model(X_batch)  # (B,1)
                loss = F.binary_cross_entropy(preds, y_batch)
                loss.backward()
                optimizer.step()

                bs = X_batch.size(0)
                with torch.no_grad():
                    predicted = (preds >= 0.5).float()
                    acc = (predicted == y_batch).float().mean().item()

                train_loss += loss.item() * bs
                train_acc += acc * bs
                n_train += bs

            train_loss /= n_train
            train_acc /= n_train

            metrics_train = self.eval(model, train_set, device)
            metrics_val = self.eval(model, val_set, device)
            val_acc = metrics_val["accuracy"]

            print(
                f"[{model_name}] Epoch {epoch+1:02d} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Eval Train Acc: {metrics_train['accuracy']:.4f} | "
                f"Eval Val Acc: {metrics_val['accuracy']:.4f}"
            )

            # Early stopping + sauvegarde meilleurs poids
            if val_acc > best_val:
                best_val = val_acc
                counter = 0
                torch.save(model.state_dict(), f"best_model_weights_{model_name}.pt")
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping for {model_name}.")
                    break

        return best_val

    @torch.inference_mode
    def eval(
        self,
        model: torch.nn.Module,
        dataset: MentalDataset,
        device: torch.device,
    ) -> Dict[str, float]:
        model.eval()

        n = len(dataset)
        n_samples = min(self.eval_samples, n)

        sampler = RandomSampler(dataset, replacement=False, num_samples=n_samples)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=0,
        )

        all_acc: List[torch.Tensor] = []
        all_loss: List[torch.Tensor] = []

        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = F.binary_cross_entropy(preds, y_batch, reduction="none")

            predicted = (preds >= 0.5).float()
            acc = (predicted == y_batch).float()

            all_acc.append(acc)
            all_loss.append(loss)

        acc_tensor = torch.cat(all_acc).mean()
        loss_tensor = torch.cat(all_loss).mean()

        return {
            "accuracy": float(acc_tensor),
            "loss": float(loss_tensor),
        }
