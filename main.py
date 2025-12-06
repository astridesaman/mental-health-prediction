from src.trainer import MentalHealthTrainer, load_and_preprocess
from src.dataset import MentalDataset
from src.model import LinearBaseline, MentalHealthModelNN

import torch
import torch.optim as optim


def main():
    # 1) Charger & prétraiter
    X_train, y_train, X_val, y_val, mean, std, feature_cols = load_and_preprocess(
        "./data/train.csv"
    )

    train_set = MentalDataset(X_train, y_train)
    val_set = MentalDataset(X_val, y_val)

    input_dim = X_train.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = MentalHealthTrainer(
        batch_size=64,
        n_epochs=25,
        eval_samples=10_000,
    )

    # 2) Définir les modèles à comparer
    models = {
        "linear": LinearBaseline(input_dim),
        "nn": MentalHealthModelNN(input_dim),
    }

    val_scores = {}

    # 3) Entraîner chaque modèle et stocker la meilleure val accuracy
    for name, model in models.items():
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best_val_acc = trainer.train(
            model=model,
            train_set=train_set,
            val_set=val_set,
            optimizer=optimizer,
            device=device,
            model_name=name,
        )
        val_scores[name] = best_val_acc

    # 4) Afficher un tableau de comparaison
    print("\n=== Model comparison (validation accuracy) ===")
    print("+---------+--------------------+")
    print("| Model   | Val Accuracy       |")
    print("+---------+--------------------+")
    for name, acc in val_scores.items():
        print(f"| {name:<7} | {acc*100:>6.2f}%            |")
    print("+---------+--------------------+")

    # 5) Déterminer le meilleur modèle
    best_model_name = max(val_scores, key=val_scores.get)
    best_val = val_scores[best_model_name]
    print(f"\nBest model: {best_model_name} ({best_val*100:.2f}% val accuracy)")

    # 6) Sauvegarder un checkpoint "global" pour predict.py
    #    On recharge les poids du meilleur modèle et on les stocke dans un seul fichier.
    best_state_dict = torch.load(f"best_model_weights_{best_model_name}.pt", map_location="cpu")
    torch.save(
        {
            "model_type": best_model_name,   # "linear" ou "nn"
            "state_dict": best_state_dict,
            "mean": mean,
            "std": std,
            "feature_cols": feature_cols,
        },
        "mental_health_model.pt",
    )
    print("Saved best model in mental_health_model.pt")


if __name__ == "__main__":
    main()
