# Challenge Data ENS #163 — Egis : Prédiction du taux de mesures invalides

> **Résultat** : Corrélation de Spearman : 0.6421 sur validation croisée 5-folds - Top 5 : 0,5699

---

## Contexte

Egis est une entreprise française d'ingénierie des infrastructures qui exploite des capteurs de comptage de trafic routier sur l'ensemble du territoire. Ces capteurs enregistrent des flux de véhicules mais produisent parfois des **mesures invalides** en raison des conditions météorologiques, de pannes matérielles, ou d'anomalies de calibration.

L'objectif est de prédire le ratio de mesures invalides (`invalid_ratio` ∈ [0, 1]) à partir de données de localisation, météorologiques et temporelles, afin d'anticiper les défaillances et optimiser la maintenance.

---

## Dataset

| Fichier | Lignes | Description |
|---|---|---|
| `x_train` | 6 076 546 | Features d'entraînement |
| `y_train` | 6 076 546 | Cible : `invalid_ratio` |
| `x_test` | 2 028 750 | Features de test |

**Features** :
- **Localisation** : `longitude_scaled`, `latitude_scaled`
- **Météo** : `Precipitations`, `HauteurNeige`, `Temperature`, `ForceVent`
- **Temporel** : `day_of_week`, `month_of_year`, `hour`
- **Volume** : `total_count` (nombre de mesures sur la période)

---

## Approche

### Métrique : Corrélation de Spearman
La corrélation de Spearman mesure la qualité de l'**ordonnancement** des prédictions, pas les valeurs exactes. On exploite cela en transformant la cible en rangs normalisés avant l'entraînement (rank regression).

### Pipeline complet

```
Données brutes
    ↓
Feature Engineering (encodage cyclique, log-transform, interactions météo)
    ↓
XGBoost en régression
    ↓
Validation croisée 5-folds (Spearman OOF)
    ↓
Optimisation Optuna (30 trials)
    ↓
Ré-entraînement avec meilleurs hyperparamètres
    ↓
Analyse SHAP + Feature Importance
    ↓
Prédictions sur x_test → soumission
```

### Quelques Features créées
| Feature | Description |
|---|---|
| `hour_sin`, `hour_cos` | Encodage cyclique de l'heure |
| `day_sin`, `day_cos` | Encodage cyclique du jour |
| `month_sin`, `month_cos` | Encodage cyclique du mois |
| `log_total_count` | Log(1 + total_count) |
| `weather_severity` | Précipitations + Neige × 0.1 |
| `wind_rain` | ForceVent × Précipitations |
| `has_snow`, `has_rain` | Indicateurs météo binaires |
| `time_period` | Nuit / Matin / Après-midi / Soir |
| `season` | Saison (hiver/printemps/été/automne) |
| `is_weekend` | Indicateur week-end |
| `temp_snow` | Température × Neige |

### Modèle : LightGBM
- **Objectif** : `regression_l1` (MAE — robuste aux outliers)
- **Astuce Spearman** : entraînement sur rangs normalisés ∈ [0,1]
- **Hyperparamètres** : optimisés avec Optuna (TPE sampler, 50 trials)
- **Validation** : KFold 5-folds avec early stopping

---

## Structure du projet

```
Egis/
├── notebooks/
│   └── egis_challenge.ipynb   # Notebook principal (EDA → Modèle → Soumission)
├── outputs/
│   ├── y_test_predictions.csv # Fichier de soumission
│   ├── target_distribution.png
│   ├── temporal_patterns.png
│   ├── geographic_patterns.png
│   ├── spearman_correlations.png
│   ├── feature_importance.png
│   ├── shap_summary.png
│   ├── optuna_convergence.png
│   └── calibration.png
├── requirements.txt
└── README.md
```

---

## Installation & Exécution

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer Jupyter
jupyter notebook notebooks/egis_challenge.ipynb
```

**Note** : Le dataset (~500 Mo) n'est pas inclus dans ce repo. Les fichiers CSV doivent être placés dans un dossier ***data***.

---

## Technologies

![Python](https://img.shields.io/badge/Python-3.10-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green)
