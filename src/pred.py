import numpy as np
import pandas as pd


# mêmes listes que dans load_and_preprocess
NUM_COLS = [
    "Age",
    "Academic Pressure",
    "Work Pressure",
    "CGPA",
    "Study Satisfaction",
    "Job Satisfaction",
    "Work/Study Hours",
    "Financial Stress",
]

CAT_COLS = [
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

DROP_COLS = ["id", "Name"]


def preprocess_test(df_test: pd.DataFrame, feature_cols, mean, std) -> np.ndarray:
    """Applique le même prétraitement que pour le train,
    puis aligne les colonnes sur feature_cols.
    """

    # on enlève les colonnes inutiles
    df = df_test.drop(columns=DROP_COLS)

    # valeurs manquantes
    df[NUM_COLS] = df[NUM_COLS].fillna(df[NUM_COLS].median())
    df[CAT_COLS] = df[CAT_COLS].fillna("Unknown")

    # one-hot
    df = pd.get_dummies(df, columns=CAT_COLS)

    # réaligner les colonnes sur celles du train
    df = df.reindex(columns=feature_cols, fill_value=0.0)

    X_test = df.astype("float32").values

    # standardisation avec mean/std du train
    X_test = (X_test - mean) / (std + 1e-8)

    return X_test