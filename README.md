# CNN Skylanders

FR : Projet personnel de vision par ordinateur pour reconnaître l’élément d’une figurine Skylanders à partir d’une image.  
EN: Personal computer vision project that predicts the element of a Skylanders figure from an image.

---

## Français

### Présentation

Ce projet a pour objectif de classifier automatiquement des figurines **Skylanders** à partir d’une image, en utilisant un **réseau de neurones convolutif (CNN)** entraîné sur un dataset d’images.  
L’application finale permet d’uploader une image et d’obtenir une prédiction du type de figurine.

Le projet se compose de trois parties principales :

- `model_architecture.py` : définition de l’architecture du réseau de neurones.
- `CNN-skylander.ipynb` : notebook pour le prétraitement, l’entraînement et l’évaluation du modèle.
- `app_skylander.py` : application Streamlit pour tester le modèle sur de nouvelles images.

### Fonctionnalités

- Classification d’images de figurines Skylanders.
- Prédiction parmi plusieurs éléments du jeu.
- Interface simple avec **Streamlit**.
- Visualisation des zones importantes de l’image grâce à **Grad-CAM**.
- Sauvegarde et rechargement du modèle entraîné.

### Structure du projet

```bash
CNN-skylanders/
│── app_skylander.py
│── CNN-skylander.ipynb
│── model_architecture.py
│── modelskylander.pth
│── encoderskylander.pkl
└── README.md

