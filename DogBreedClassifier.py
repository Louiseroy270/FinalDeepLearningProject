import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

data_path = "Images" 

target_size = (224, 224)  # Taille cible pour redimensionner les images

batch_size = 1000  # Nombre d'images par lot car sans lot, fichier trop lourd 

images = []
labels = []
breed_names = sorted(os.listdir(data_path))  # On liste les noms de dossiers de races de chiens

label_encoder = LabelEncoder()

def process_batch(batch_images, batch_labels):
    batch_images = np.array(batch_images)
    batch_labels = np.array(batch_labels)
    
    # Encodage des labels
    encoded_labels = label_encoder.fit_transform(batch_labels)
    one_hot_labels = to_categorical(encoded_labels)

    # On vérifie avec des prints que tout se passe correctement (encodage des labels et forme des images)
    print("Forme des images :", batch_images.shape)
    print("Forme des labels encodés en one-hot :", one_hot_labels.shape)

    # On divise les données en ensembles d'entraînement et de test (80% / 20%)
    train_images, test_images, train_labels, test_labels = train_test_split(images, one_hot_labels, test_size=0.2, random_state=42)

    # On divise l'ensemble d'entraînement en ensembles d'entraînement et de validation (80% / 20%)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

    # On vérifie la taille des ensembles voir si tout est bon 
    print("Taille de l'ensemble d'entraînement :", len(train_images))
    print("Taille de l'ensemble de validation :", len(val_images))
    print("Taille de l'ensemble de test :", len(test_images))
    

# La boucle parcourt les dossiers de races de chiens et charge les images par lot
for i, breed in enumerate(breed_names):
    breed_folder = os.path.join(data_path, breed)
    if os.path.isdir(breed_folder):
        print(f"Chargement des images de {breed}")
        batch_images = []
        batch_labels = []
        for img_file in os.listdir(breed_folder):
            img_path = os.path.join(breed_folder, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, target_size)  # Redimensionne les images à la taille cible
                img = img / 255.0  # Normalise les valeurs des pixels entre 0 et 1
                batch_images.append(img)
                batch_labels.append(breed)
                
                # Si la taille du lot est atteinte, le lot est traité ici
                if len(batch_images) >= batch_size:
                    process_batch(batch_images, batch_labels)
                    batch_images = []
                    batch_labels = []

# On traite les images restantes qui ne constituent pas un lot complet
if len(batch_images) > 0:
    process_batch(batch_images, batch_labels)