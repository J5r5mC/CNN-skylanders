# CNN Skylanders

> **FR** — Réseau de neurones convolutif pour identifier l'élément d'une figurine Skylanders à partir d'une image.  
> **EN** — Convolutional neural network to identify the element of a Skylanders figure from an image.

---

## Français

### Présentation

Ce projet personnel de **vision par ordinateur** utilise un réseau de neurones convolutif (CNN) pour reconnaître l'**élément** d'une figurine **Skylanders** (Feu, Eau, Terre, Air, etc.) à partir d'une simple photo.

Il constitue un pipeline complet : collecte des images, entraînement du modèle, évaluation des performances, et déploiement dans une interface interactive.

---

### Structure du projet

```
CNN-skylanders/
├── Images/                  # Dataset : images organisées par élément
├── Prediction/              # Images de test pour l'inférence
├── CNN-skylander.ipynb      # Notebook principal (données, entraînement, évaluation)
├── model_architecture.py    # Définition de l'architecture CNN
├── app_skylander.py         # Application Streamlit pour l'inférence
├── model_skylander.pth      # Poids du modèle entraîné (PyTorch)
├── encoder_skylander.pkl    # Encodeur des labels (classes)
└── README.md
```

---

### Fonctionnalités

- **Classification** d'images de figurines Skylanders par élément
- **Entraînement** du CNN sur un dataset d'images personnalisé
- **Visualisation Grad-CAM** pour interpréter les zones d'activation du modèle
- **Application Streamlit** interactive pour tester le modèle sur de nouvelles images
- **Sauvegarde et chargement** du modèle entraîné (`.pth`) et de l'encodeur (`.pkl`)

---

### Utilisation

#### 1. Cloner le dépôt

```bash
git clone https://github.com/J5r5mC/CNN-skylanders.git
cd CNN-skylanders
```

#### 2. Installer les dépendances

```bash
pip install torch torchvision streamlit scikit-learn pillow
```

#### 3. Entraîner le modèle

Ouvrir et exécuter le notebook `CNN-skylander.ipynb` pour préparer les données et entraîner le modèle.

#### 4. Lancer l'application

```bash
streamlit run app_skylander.py
```

Uploader une image de figurine Skylanders dans l'interface, et le modèle prédit son élément.

---

### Architecture

Le modèle est un CNN défini dans `model_architecture.py`, entraîné avec **PyTorch**. Il prend en entrée une image de figurine et retourne la classe d'élément prédite parmi les différentes factions du jeu.

---

## English

### Overview

This personal **computer vision** project uses a Convolutional Neural Network (CNN) to recognize the **element** of a **Skylanders** figure (Fire, Water, Earth, Air, etc.) from a photo.

It covers a complete pipeline: image collection, model training, performance evaluation, and deployment via an interactive web interface.

---

### Project Structure

```
CNN-skylanders/
├── Images/                  # Dataset: images organized by element
├── Prediction/              # Test images for inference
├── CNN-skylander.ipynb      # Main notebook (data prep, training, evaluation)
├── model_architecture.py    # CNN architecture definition
├── app_skylander.py         # Streamlit app for inference
├── model_skylander.pth      # Trained model weights (PyTorch)
├── encoder_skylander.pkl    # Label encoder (classes)
└── README.md
```

---

### Features

- **Classification** of Skylanders figures by element type
- **CNN training** on a custom image dataset
- **Grad-CAM visualization** to interpret the model's activation zones
- **Interactive Streamlit app** to test the model on new images
- **Model persistence** with saved weights (`.pth`) and label encoder (`.pkl`)

---

### Usage

#### 1. Clone the repository

```bash
git clone https://github.com/J5r5mC/CNN-skylanders.git
cd CNN-skylanders
```

#### 2. Install dependencies

```bash
pip install torch torchvision streamlit scikit-learn pillow
```

#### 3. Train the model

Open and run the `CNN-skylander.ipynb` notebook to prepare the data and train the model.

#### 4. Launch the app

```bash
streamlit run app_skylander.py
```

Upload a Skylanders figure image in the interface, and the model will predict its element.

---

### Architecture

The model is a CNN defined in `model_architecture.py`, trained with **PyTorch**. It takes a figure image as input and outputs the predicted element class among the different factions of the game.

---

### Tech Stack

| Tool | Usage |
|------|-------|
| Python | Core language |
| PyTorch | Model training & inference |
| Streamlit | Web interface |
| scikit-learn | Label encoding |
| Pillow | Image processing |
| Jupyter Notebook | Experimentation & training |
