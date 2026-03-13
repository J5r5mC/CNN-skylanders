# CNN Skylanders

FR : Projet personnel de vision par ordinateur pour reconnaître l’élément d’une figurine Skylanders à partir d’une image.  
EN: Personal computer vision project to recognize the element of a Skylanders figure from an image.

---

## Français

### Présentation

Ce projet personnel utilise un **réseau de neurones convolutif (CNN)** pour reconnaître l’élément d’une figurine **Skylanders** à partir d’une image.

Le dépôt contient :
- un dossier `Images/` avec les images utilisées pour l’entraînement ;
- un dossier `prediction/` avec d’autres images pour tester le modèle ;
- un notebook pour préparer les données, entraîner et évaluer le réseau ;
- une application **Streamlit** pour utiliser le modèle sur de nouvelles images.

L’objectif est de proposer un projet complet de vision par ordinateur, depuis la préparation des données jusqu’à l’inférence dans une interface simple.

### Fonctionnalités

- Classification d’images de figurines Skylanders.
- Prédiction de l’élément de la figurine.
- Interface simple et interactive avec **Streamlit**.
- Visualisation **Grad-CAM** pour montrer les zones importantes de l’image.
- Sauvegarde et chargement du modèle entraîné.

### Structure du projet

```bash
CNN-skylanders/
│── Images/                   # Images utilisées pour l'entraînement
│── prediction/               # Images de test pour essayer la prédiction
│── app_skylander.py          # Application Streamlit pour l'inférence
│── CNN-skylander.ipynb       # Notebook de préparation, entraînement et évaluation
│── model_architecture.py     # Architecture du CNN
│── modelskylander.pth        # Poids du modèle entraîné
│── encoderskylander.pkl      # Encodeur des labels
└── README.md
