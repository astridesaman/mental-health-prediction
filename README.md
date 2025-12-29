# Mental Health Prediction 🧠 – Python / PyTorch

# Prédiction de la Dépression – Projet Kaggle (Python)

Projet réalisé en **Python** dans le cadre du cours de **Python pour l’Intelligence Artificielle** en **Licence 3 Intelligence Artificielle (L3 IA)** à l’**Université Côte d’Azur**.  
L’objectif est de prédire la **probabilité de dépression** d’une personne à partir de données personnelles, académiques et professionnelles, issues d’une compétition Kaggle (*Playground Series – Exploring Mental Health Data*).

This project was developed in **Python** as part of the **Python for AI** course in the **3rd year of the Artificial Intelligence Bachelor’s degree (L3 AI)** at **Université Côte d'Azur**.  
The goal is to predict the **probability of depression** for an individual based on personal, academic and professional features, using data from a Kaggle playground competition (*Exploring Mental Health Data*).

---

## 📌 Objectifs du projet

- Charger et prétraiter les données Kaggle :
  - Suppression des colonnes non pertinentes (`id`, `Name`)
  - Gestion des valeurs manquantes
  - Encodage one-hot des variables catégorielles
  - Standardisation des variables numériques
- Mettre en place un pipeline complet d’apprentissage supervisé en Python / PyTorch.
- Entraîner et comparer deux modèles de classification :
  - **Modèle linéaire** (régression logistique / Linear baseline)  
  - **Réseau de neurones fully-connected** (MLP avec 2 couches cachées, Dropout, régularisation L2)
- Évaluer les modèles via :
  - Loss (Binary Cross Entropy)  
  - Accuracy (train / validation)
- Sélectionner le meilleur modèle sur l’ensemble de validation.
- Générer un fichier `submission.csv` prêt à être soumis sur Kaggle.

## 📌 Project Objectives

- Load and preprocess the Kaggle dataset:
  - Remove non-informative columns (`id`, `Name`)
  - Handle missing values
  - One-hot encode categorical variables
  - Standardize numerical features
- Implement a complete supervised learning pipeline in Python / PyTorch.
- Train and compare two classification models:
  - **Linear model** (logistic regression / Linear baseline)  
  - **Fully-connected neural network** (MLP with 2 hidden layers, Dropout, L2 regularization)
- Evaluate models using:
  - Binary Cross Entropy loss  
  - Accuracy (train / validation)
- Select the best-performing model on the validation set.
- Generate a `submission.csv` file for Kaggle submission.

---

## 📂 Données utilisées

Données issues de la compétition Kaggle *Exploring Mental Health Data* (Playground Series – S4, E11) :

- **train.csv**  
  Contient les observations annotées avec la variable cible `Depression`.

- **test.csv**  
  Même structure que `train.csv` mais sans la colonne `Depression`. Utilisé pour produire la soumission Kaggle.

- **sample_submission.csv**  
  Fichier gabarit contenant les colonnes `id` et `Depression`, indiquant le format attendu pour `submission.csv`.

## 📂 Data Used

Data comes from the Kaggle *Exploring Mental Health Data* playground competition:

- **train.csv**  
  Includes all training samples with the target variable `Depression`.

- **test.csv**  
  Same structure as `train.csv` but without the `Depression` column. Used for generating predictions.

- **sample_submission.csv**  
  Template file with columns `id` and `Depression`, defining the expected format for `submission.csv`.

---

## 🧠 Technologies

- Python 3  
- pandas, numpy  
- PyTorch (torch, torch.nn, DataLoader)  
- scikit-learn (pour certaines métriques / split éventuel)  
- Kaggle (plateforme d’évaluation)

---

## 🏗️ Structure du projet / Project Structure

```bash
mental-health-prediction/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── src/
│   ├── __init__.py
│   ├── dataset.py        # Classe MentalDataset (PyTorch Dataset)
│   ├── model.py          # LinearBaseline + MentalHealthModelNN + MentalHealthModelNNv2
│   └── trainer.py        # load_and_preprocess + MentalHealthTrainer
│
├── main.py               # Entraînement + comparaison des modèles
└── README.md
```
### 👩‍💻 Auteurs

- Projet réalisé par : Astride SAMAN
- Licence Informatique 3 – Intelligence Artificielle
- Université Côte d’Azur
