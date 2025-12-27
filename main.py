import torch
import torch.optim as optim

from src.trainer import (
    MentalHealthTrainer,
    load_and_preprocess,
    predict_on_test,
    generate_kaggle_submission,
)
from src.dataset import MentalDataset
from src.model import LinearBaseline, MentalHealthModelNN, MentalHealthModelNNv2


def build_model(name: str, input_dim: int):
    if name == "linear":
        return LinearBaseline(input_dim)
    if name == "nn":
        return MentalHealthModelNN(input_dim)
    if name == "nn2":
        return MentalHealthModelNNv2(input_dim)
    raise ValueError(f"Unknown model name: {name}")


def build_optimizer(model: torch.nn.Module, model_name: str):
    """
    Choix des hyperparamètres (lr) issus du tuning :
      - linear : 5e-3
      - nn     : 5e-4
      - nn2    : 1e-3
    weight_decay gardé constant à 1e-4 pour régularisation L2.
    """
    best_lrs = {
        "linear": 5e-3,
        "nn": 5e-4,
        "nn2": 1e-3,
    }
    lr = best_lrs[model_name]
    return optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)


def main():
    # 1) Prétraitement
    X_train, y_train, X_val, y_val, mean, std, feature_cols = load_and_preprocess("./data/train.csv")
    train_set = MentalDataset(X_train, y_train)
    val_set = MentalDataset(X_val, y_val)

    input_dim = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = MentalHealthTrainer(batch_size=64, n_epochs=25, eval_samples=10_000)

    # 2) Modèles à comparer
    model_names = ["linear", "nn", "nn2"]
    val_scores = {}

    # 3) Entraîner chaque modèle avec son meilleur lr
    for name in model_names:
        model = build_model(name, input_dim)
        optimizer = build_optimizer(model, name)

        best_val_acc = trainer.train(
            model=model,
            train_set=train_set,
            val_set=val_set,
            optimizer=optimizer,
            device=device,
            model_name=name,
        )
        val_scores[name] = best_val_acc

    # 4) Tableau comparaison
    print("\n=== Model comparison (validation accuracy) ===")
    print("+---------+--------------------+----------------+")
    print("| Model   | Best LR            | Val Accuracy   |")
    print("+---------+--------------------+----------------+")
    best_lrs = {"linear": 5e-3, "nn": 5e-4, "nn2": 1e-3}
    for name, acc in val_scores.items():
        print(f"| {name:<7} | {best_lrs[name]:<18} | {acc*100:>6.2f}%       |")
    print("+---------+--------------------+----------------+")

    # 5) Choisir meilleur modèle
    best_model_name = max(val_scores, key=val_scores.get)
    best_val = val_scores[best_model_name]
    print(f"\nBest model: {best_model_name} ({best_val*100:.2f}% val accuracy)")

    # 6) Sauvegarder checkpoint global
    best_state_dict = torch.load(f"best_model_weights_{best_model_name}.pt", map_location="cpu")
    torch.save(
        {
            "model_type": best_model_name,
            "state_dict": best_state_dict,
            "mean": mean,
            "std": std,
            "feature_cols": feature_cols,
            "input_dim": input_dim,
            "best_lr": best_lrs[best_model_name],
        },
        "mental_health_model.pt",
    )
    print("Saved best model in mental_health_model.pt")

    # 7) Recréer meilleur modèle + charger poids
    best_model = build_model(best_model_name, input_dim)
    best_model.load_state_dict(torch.load(f"best_model_weights_{best_model_name}.pt", map_location=device))
    best_model.to(device)
    best_model.eval()

    # 8) Prédire sur test.csv  
    ids, probas, preds = predict_on_test(
        model=best_model,
        test_csv_path="./data/test.csv",
        feature_cols=feature_cols,
        mean=mean,
        std=std,
        device=device,
    )

    # 9) Générer fichiers CSV
    # generate_csv_probabilities("predictions.csv", ids, probas)  # si tu veux un fichier avec probas
    generate_kaggle_submission("submission.csv", ids, preds)
    print("Generated submission.csv")


if __name__ == "__main__":
    main()
