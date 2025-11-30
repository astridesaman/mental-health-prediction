from dataclasses import dataclass
from typing import Dict, List
from src.dataset import MentalDataset
from src.model import MentalHealthModel

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler
import torch.optim as optim


# 1. Chargement & prétraitement des données

def load_and_preprocess(path: str):
    """
    Charge train.csv, applique le prétraitement et renvoie :
    X_train, y_train, X_val, y_val, mean, std, feature_cols
    """

    df = pd.read_csv(path)

    # Cible
    target_col = "Depression"

    # Colonnes inutiles
    df = df.drop(columns=["id", "Name"])

    # Colonnes numériques
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

    # Colonnes catégorielles
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

    # Séparer X / y
    y = df[target_col].astype("float32").values
    X = df.drop(columns=[target_col])

    # Valeurs manquantes
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    X[cat_cols] = X[cat_cols].fillna("Unknown")

    # One-hot encoding
    X = pd.get_dummies(X, columns=cat_cols)

    # Garder la liste des colonnes (pour test plus tard)
    feature_cols = X.columns.tolist()

    # Conversion en numpy
    X = X.astype("float32").values

    # Split train / val (80 / 20)
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))

    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Standardisation
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    return X_train, y_train, X_val, y_val, mean, std, feature_cols

# 4. Trainer 

@dataclass
class MentalHealthTrainer:
    batch_size: int
    n_epochs: int
    eval_samples: int  # nb d'échantillons pour l'éval (train/val)

    def train(
        self,
        model: MentalHealthModel,
        train_set: MentalDataset,
        val_set: MentalDataset,
        optimizer: optim.Optimizer,
        device: torch.device,
    ):
        model.to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {n_params:,}")

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
            # TRAIN 
            model.train()
            train_loss = 0.0
            train_acc = 0.0
            n_train = 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                preds = model(X_batch)  # (B,1) dans [0,1]
                loss = F.binary_cross_entropy(preds, y_batch)
                loss.backward()
                optimizer.step()

                # stats
                with torch.no_grad():
                    predicted = (preds >= 0.5).float()
                    acc = (predicted == y_batch).float().mean().item()

                bs = X_batch.size(0)
                train_loss += loss.item() * bs
                train_acc += acc * bs
                n_train += bs

            train_loss /= n_train
            train_acc /= n_train

            # EVAL TRAIN / VAL 
            metrics_train = self.eval(model, train_set, device)
            metrics_val = self.eval(model, val_set, device)

            val_acc = metrics_val["accuracy"]

            print(
                f"[Epoch {epoch+1:02d}] "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Eval Train Acc: {metrics_train['accuracy']:.4f} | "
                f"Eval Val Acc: {metrics_val['accuracy']:.4f}"
            )

            # Early stopping simple
            if val_acc > best_val:
                best_val = val_acc
                counter = 0
                torch.save(model.state_dict(), "best_model_weights.pt")
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping déclenché.")
                    break

        # Recharger les meilleurs poids
        model.load_state_dict(torch.load("best_model_weights.pt", map_location=device))

    @torch.inference_mode
    def eval(
        self,
        model: MentalHealthModel,
        dataset: MentalDataset,
        device: torch.device,
    ) -> Dict[str, float]:
        model.eval()

        n = len(dataset)
        n_samples = min(self.eval_samples, n)

        sampler = RandomSampler(
            dataset,
            replacement=False,
            num_samples=n_samples,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=0,
        )

        all_acc: List[Tensor] = []
        all_loss: List[Tensor] = []

        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            preds = model(X_batch)  # (B,1)
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

def main():
    # 1) Charger & prétraiter
    X_train, y_train, X_val, y_val, mean, std, feature_cols = load_and_preprocess(
        "./data/train.csv"
    )

    train_set = MentalDataset(X_train, y_train)
    val_set = MentalDataset(X_val, y_val)

    input_dim = X_train.shape[1]
    model = MentalHealthModel(input_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    trainer = MentalHealthTrainer(
        batch_size=64,
        n_epochs=25,
        eval_samples=10_000,  # tu peux augmenter
    )

    trainer.train(model, train_set, val_set, optimizer, device)

    # Sauvegarde du modèle + stats (pour predict.py plus tard)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "feature_cols": feature_cols,
        },
        "mental_health_model.pt",
    )

    print("Modèle + stats sauvegardés dans mental_health_model.pt")


if __name__ == "__main__":
    main()

