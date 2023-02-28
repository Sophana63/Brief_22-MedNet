# Brief 22 - MedNet

## Description du code Flask pour prédiction d'images médicales

Ce code utilise le framework Flask pour créer une application web qui permet de prédire la catégorie d'une image médicale téléchargée par l'utilisateur.

L'application utilise un modèle de réseau de neurones pré-entraîné sur un ensemble de données médicales. Le modèle est chargé à partir d'un fichier de sauvegarde et utilisé pour prédire la catégorie de chaque image téléchargée par l'utilisateur.

## Structure du code

Le code est organisé en plusieurs fichiers et répertoires:

### Fichiers Python

- `MedNet.py`: définit le modèle de réseau de neurones utilisé pour la classification d'images médicales.
- `app.py`: définit l'application Flask et les différentes routes pour accéder aux pages web de l'application.

### Répertoires

- `static/uploads`: contient les fichiers image téléchargés par les utilisateurs.

- `static/saved_model`: contient le modèle de réseau de neurones pré-entraîné.

- `static/readme`: contient le fichier README.md pour la documentation de l'application.

- `templates`: contient les fichiers HTML pour les différentes pages web de l'application.

## Installation et utilisation

Le code est écrit en Python 3 et utilise plusieurs bibliothèques Python, notamment Flask, numpy, matplotlib, PIL, torch et torchvision.

Pour installer les bibliothèques requises, exécutez la commande suivante :

```
pip install -r requirements.txt
```
Lien vers le fichier [requirements.txt]()
Pour lancer l'application, exécutez la commande suivante :

```
python index.py
```

Une fois l'application en cours d'exécution, vous pouvez accéder à l'interface web en ouvrant votre navigateur et en accédant à l'adresse suivante : [http://localhost:5000/](http://localhost:5000/)

## Routes de l'application

L'application dispose de plusieurs routes pour accéder aux différentes pages web :

- `/` : page d'accueil de l'application.
- `/upload` : page de téléchargement de l'image.
- `/predict` : page de prédiction de l'image.
- `/notebook` : page pour accéder à la documentation de l'application.
- `/explication` : page pour expliquer le fonctionnement de l'application.