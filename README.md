# 🫁 Détection de Pneumonie — CNN from Scratch (PyTorch)

> Classification binaire de radiographies thoraciques (NORMAL vs PNEUMONIA) à l'aide d'un réseau de neurones convolutif construit entièrement from scratch avec PyTorch.

---

## 📌 Présentation

Ce projet implémente un CNN personnalisé pour la détection automatique de pneumonie à partir de radiographies thoraciques. Il a été réalisé dans le cadre du **TP7** d'un cours de Deep Learning, en suivant une approche pédagogique rigoureuse : aucun modèle pré-entraîné (VGG, ResNet, etc.) n'est utilisé.

Le projet comprend deux notebooks complémentaires :

| Notebook | Description |
|---|---|
| `TP7_01.ipynb` | Version principale : architecture CNN, data augmentation, entraînement avec Early Stopping, évaluation complète |
| `CNN_scratch2_TP_style.ipynb` | Version améliorée : validation stratifiée reconstruite depuis le train, normalisation calculée sur les données, corrections de bugs grayscale |

---

## 📂 Structure du Dataset

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

> **Source :** [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
> Le dataset est déséquilibré (~3× plus de cas PNEUMONIA que NORMAL), ce qui est pris en compte via les poids de classe.

---

## 🏗️ Architecture du Modèle

```
Input (1×128×128 ou 1×160×160)
    │
    ├── Conv Block 1 : Conv2d(1→32, k=3, p=1) + BN + ReLU + MaxPool(2)
    ├── Conv Block 2 : Conv2d(32→64, k=3, p=1) + BN + ReLU + MaxPool(2)
    ├── Conv Block 3 : Conv2d(64→128, k=3, p=1) + BN + ReLU + MaxPool(2)
    ├── Conv Block 4 : Conv2d(128→256, k=3, p=1) + BN + ReLU + MaxPool(2)
    │
    ├── Global Average Pooling (256×1×1)  ← réduit x64 vs Flatten
    │
    └── Classifier : Linear(256→128) + ReLU + Dropout(0.4) + Linear(128→2)
```

**Techniques anti-overfitting :**
- Batch Normalization après chaque couche convolutive
- Global Average Pooling (256 paramètres en entrée du FC vs 16 384 avec Flatten)
- Dropout(0.4) dans le classifieur
- Weight Decay (L2) dans l'optimiseur

---

## ⚙️ Hyperparamètres

| Paramètre | Valeur | Justification |
|---|---|---|
| Optimiseur | Adam | Adaptatif, convergence rapide |
| Learning Rate | 1e-3 / 3e-4 | Standard pour Adam |
| Weight Decay | 1e-4 | Régularisation L2 légère |
| Scheduler | ReduceLROnPlateau | Réduit le LR si val_loss stagne |
| Batch Size | 32 | Bon compromis mémoire/stabilité |
| Image Size | 128 / 160 | Suffisant pour la classification |
| Epochs max | 30 | Arrêt anticipé par Early Stopping |
| Patience | 7 | Avant Early Stopping |
| Init poids | Kaiming Uniform | Adaptée aux activations ReLU |

---

## 🔄 Data Augmentation

Appliquée uniquement sur le set d'entraînement :

- `RandomHorizontalFlip` — positions miroir réalistes en radiologie
- `RandomRotation(10°)` + `RandomAffine` — légères distorsions géométriques
- `ColorJitter` — variations de contraste et luminosité
- `RandomErasing` — masquage aléatoire (régularisation) — placé **après** `ToTensor`
- `Normalize` — centrage/réduction (calculé sur le train dans la v2)

---

## 🚀 Utilisation

### Prérequis

```bash
pip install torch torchvision scikit-learn matplotlib seaborn pandas pillow
```

### Lancer sur Google Colab

1. Monter Google Drive et placer le dataset au chemin attendu :
   ```
   /content/drive/MyDrive/TP7+projet/chest_xray/
   ```
2. Ouvrir le notebook souhaité dans Colab
3. Activer le GPU : `Exécution > Modifier le type d'exécution > GPU`
4. Exécuter toutes les cellules

---

## 📊 Résultats

Le modèle est évalué sur :
- **Accuracy** (train, validation, test)
- **Matrice de confusion**
- **Rapport de classification** (précision, rappel, F1 par classe)
- **Courbes loss/accuracy** par époque
- **Analyse des erreurs** (visualisation des images mal classées)

---

## 📁 Fichiers générés

```
chest_xray_results_from_scratch/
├── models/
│   └── best_model_cnn_from_scratch.pt   ← meilleur checkpoint
└── results/
    ├── global_results_cnn_from_scratch.csv
    └── summary_cnn_from_scratch.txt
```

---

## 🧪 Fonctionnalités notables

- **Early Stopping** avec sauvegarde automatique du meilleur modèle
- **Poids de classe** pour corriger le déséquilibre NORMAL/PNEUMONIA
- **Gradient clipping** (`max_norm=1.0`) pour stabiliser l'entraînement
- **Validation stratifiée** reconstruite depuis le train (v2) pour pallier la petitesse du dossier `val` d'origine
- **Normalisation calculée** sur les données d'entraînement (plus précise que `Normalize(0.5, 0.5)`)
- **Prédiction sur image seule** avec affichage de la confiance

---

## 🛠️ Stack Technique

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![Colab](https://img.shields.io/badge/Google%20Colab-compatible-orange?logo=googlecolab)
![scikit-learn](https://img.shields.io/badge/scikit--learn-used-yellowgreen?logo=scikitlearn)

- **PyTorch** — modèle, entraînement, inférence
- **torchvision** — transformations, chargement de données
- **scikit-learn** — split stratifié, métriques, rapport de classification
- **matplotlib / seaborn** — visualisations
- **Google Colab** — environnement d'exécution avec GPU

---

## 👤 Auteur

Projet réalisé dans le cadre d'un TP de Deep Learning (TP7).
