import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

data_path = "Images"  # Met le chemin vers ton dossier principal

images = []
labels = []

breed_names = sorted(os.listdir(data_path))  # Liste les noms de dossiers de races de chiens

for i, breed in enumerate(breed_names):
    breed_folder = os.path.join(data_path, breed)
    if os.path.isdir(breed_folder):
        print(f"Chargement des images de {breed}")
        for img_file in os.listdir(breed_folder):
            img_path = os.path.join(breed_folder, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))  # Redimensionne les images à une taille standard
            img = img / 255.0  # Normalise les valeurs des pixels entre 0 et 1
            images.append(img)
            labels.append(breed)

images = np.array(images)
labels = np.array(labels)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
one_hot_labels = to_categorical(encoded_labels)

print("Forme des images :", images.shape)
print("Forme des labels encodés en one-hot :", one_hot_labels.shape)