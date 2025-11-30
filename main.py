from src.trainer import MentalHealthTrainer, load_and_preprocess
from src.dataset import MentalDataset
from src.model import MentalHealthModel

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
    model = MentalHealthModel(input_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    trainer = MentalHealthTrainer(
        batch_size=64,
        n_epochs=25,
        eval_samples=10_000,  # possibilité d'augmenter
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
