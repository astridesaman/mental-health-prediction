# Mental Health Prediction 🧠  
*Projet de classification binaire en intelligence artificielle*

---

## 🎯 Objectif du projet

Ce projet vise à prédire si une personne présente un risque de dépression (`Depression = 1`) ou non (`Depression = 0`) à partir de caractéristiques personnelles, académiques, professionnelles et sociales.

L’objectif est double :

- Appliquer un pipeline complet de Machine Learning avec PyTorch
- Construire un modèle performant, interprétable et reproductible
- Soumettre les prédictions dans une compétition Kaggle réelle

---

## 📁 Arborescence du projet


---

## 🧠 Description des données

Le jeu de données contient :

### Variables numériques :
- Âge
- Pression académique et professionnelle
- Note moyenne (CGPA)
- Satisfaction (études / travail)
- Nombre d’heures de travail / études
- Niveau de stress financier

### Variables catégorielles :
- Genre
- Ville
- Statut (étudiant / professionnel)
- Profession
- Qualité du sommeil
- Habitudes alimentaires
- Diplôme
- Pensées suicidaires (oui / non)
- Antécédents familiaux de troubles mentaux

### Variable cible :
- `Depression` : 0 = non dépressif, 1 = dépressif

---

## ⚙️ Pipeline de traitement

### 1. Prétraitement
- Suppression des colonnes inutiles (`id`, `Name`)
- Remplissage des valeurs manquantes :
  - médiane pour les variables numériques
  - `"Unknown"` pour les variables catégorielles
- Encodage one-hot des variables catégorielles
- Normalisation des variables numériques (standardisation)
- Séparation train / validation : 80% / 20%

---

### 2. Modèle utilisé

Architecture de réseau de neurones (MLP) :

Input → Linear(128) → ReLU → Dropout
→ Linear(64) → ReLU → Dropout
→ Linear(1) → Sigmoid


- Nombre de paramètres : **53 889**
- Fonction de perte : `Binary Cross Entropy`
- Optimiseur : Adam + régularisation L2 (`weight_decay`)
- Batch size : 64
- Fonction de décision : seuil à 0.5

---

### 3. Entraînement

- Métrique principale : Accuracy
- Early stopping activé pour éviter l’overfitting
- Sauvegarde automatique des meilleurs poids

---

## 📊 Résultats obtenus

Sur l’ensemble de validation :

- **Accuracy Validation ≈ 94.3%**
- Très faible overfitting
- Bonne généralisation
- Courbe de convergence stable

Sur Kaggle :
- Score public proche de **94%**
- Performance solide pour un projet académique en IA

---

## 🔍 Problèmes rencontrés et solutions

### 1. Problèmes d'import des modules
Résolu en structurant le projet avec `src/` et `__init__.py`

### 2. Valeurs manquantes
Résolu via imputation automatique

### 3. Catégories inconnues dans test
Résolu par réalignement des colonnes avec `reindex()`

### 4. Sur-apprentissage
Résolu par :
- Dropout
- Régularisation L2
- Early stopping

---

## Comment lancer le projet

### Entraîner le modèle :

```bash
python main.py
```

### ✅ Technologies utilisées

- Python 3.13

- PyTorch

- pandas / numpy

- VS Code

- Kaggle

### ✨ Conclusion

Ce projet démontre la capacité à :

- Construire un pipeline ML complet

- Gérer des données réelles

- Implémenter un modèle de deep learning

- Évaluer et régulariser un réseau neuronal

- Déployer un modèle sur Kaggle

Il constitue une base solide pour des projets plus avancés en :

- IA médicale

- Data science

- Machine Learning appliqué à la santé

### 👩‍💻 Auteur

- Projet réalisé par : Astride SAMAN et Aya BOUROUISSE
- Licence Informatique 3 – Intelligence Artificielle
- Université Côte d’Azur