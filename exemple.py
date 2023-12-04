import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

data_path = "Images"
target_size = (224, 224)
batch_size = 32

# Créer un générateur d'images
datagen = ImageDataGenerator(rescale=1./255)
image_generator = datagen.flow_from_directory(
    data_path,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# Obtenez le nombre total d'images
num_images = len(image_generator.filenames)

# Récupérer les labels
label_encoder = LabelEncoder()
labels = image_generator.classes
label_encoder.fit(labels)
num_classes = len(label_encoder.classes_)

# Diviser les données en ensembles d'entraînement, de validation et de test
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

train_size = int(num_images * train_ratio)
val_size = int(num_images * val_ratio)

train_data = []
val_data = []
test_data = []

for i in range(num_images // batch_size):
    batch_images, batch_labels = image_generator.next()
    if len(train_data) < train_size:
        train_data.extend(batch_images)
    elif len(val_data) < val_size:
        val_data.extend(batch_images)
    else:
        test_data.extend(batch_images)

train_images = np.array(train_data)
val_images = np.array(val_data)
test_images = np.array(test_data)

train_labels = label_encoder.transform(labels[:len(train_images)])
val_labels = label_encoder.transform(labels[len(train_images):len(train_images) + len(val_images)])
test_labels = label_encoder.transform(labels[len(train_images) + len(val_images):])

train_labels = to_categorical(train_labels, num_classes=num_classes)
val_labels = to_categorical(val_labels, num_classes=num_classes)
test_labels = to_categorical(test_labels, num_classes=num_classes)

print("Taille de l'ensemble d'entraînement :", len(train_images))
print("Taille de l'ensemble de validation :", len(val_images))
print("Taille de l'ensemble de test :", len(test_images))

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Création du modèle CNN
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compilation du modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(val_images, val_labels))

# Évaluation du modèle sur l'ensemble de test
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Précision sur l'ensemble de test :", test_accuracy)