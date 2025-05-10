# Momentum Picks

## Description
Momentum Picks est une application de stratégie de trading basée sur le momentum. Cette stratégie consiste à identifier et investir dans des actions qui ont connu une forte augmentation de prix récemment, partant du principe que ces actions ont tendance à continuer leur progression à court terme.

## Fonctionnalités
- Analyse des actions du S&P 500 ou d'autres indices
- Calcul du score de momentum basé sur plusieurs périodes (1 mois, 3 mois, 6 mois, 1 an)
- Classement des actions selon leur score de momentum
- Génération de recommandations d'achat pour un portefeuille équipondéré
- Visualisation des performances historiques

## Structure du projet
```
momentum-picks/
├── data/               # Données brutes et traitées
├── src/                # Code source
│   ├── data/           # Modules de collecte et traitement des données
│   ├── analysis/       # Modules d'analyse et calcul de momentum
│   ├── visualization/  # Modules de visualisation
│   └── utils/          # Fonctions utilitaires
├── notebooks/          # Notebooks Jupyter pour l'exploration et démonstration
├── tests/              # Tests unitaires
├── requirements.txt    # Dépendances du projet
└── README.md           # Documentation du projet
```

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/kyac99/momentum-picks.git
cd momentum-picks

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation
À venir...

## Licence
Ce projet est distribué sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.
