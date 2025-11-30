from src.trainer import load_and_preprocess
from src.model import MentalHealthModel
from src.pred import preprocess_test

import torch

import pandas as pd

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Recharger le prétraitement du train
    # (on ne garde que mean, std, feature_cols)
    X_train, y_train, X_val, y_val, mean, std, feature_cols = load_and_preprocess(
        "./data/train.csv"
    )
    input_dim = X_train.shape[1]

    # 2) Charger test.csv et sample_submission.csv
    df_test = pd.read_csv("./data/test.csv")
    df_sample = pd.read_csv("./data/sample_submission.csv")

    # 3) Prétraiter X_test
    X_test = preprocess_test(df_test, feature_cols, mean, std)

    # 4) Construire le modèle et charger les meilleurs poids
    model = MentalHealthModel(input_dim)
    state_dict = torch.load("best_model_weights.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 5) Prédire
    with torch.inference_mode():
        X_tensor = torch.from_numpy(X_test).to(device)
        probs = model(X_tensor).cpu().numpy().reshape(-1)  # sigmoid déjà dans le modèle
        preds = (probs >= 0.5).astype(int)                 # 0 / 1

    # 6) Construire submission.csv
    submission = df_sample.copy()
    submission["Depression"] = preds

    submission.to_csv("submission.csv", index=False)
    print("submission.csv généré.")


if __name__ == "__main__":
    main()
