import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Chemin vers le répertoire principal contenant les dossiers d'images par classe
data_dir = 'Images'

# Définir les paramètres de prétraitement et de chargement des images
batch_size = 32
image_size = (224, 224)  # Redimensionner les images à la taille requise

# Utiliser ImageDataGenerator pour charger et prétraiter les images
datagen = ImageDataGenerator(
    rescale=1./255,  # Mise à l'échelle des valeurs de pixel entre 0 et 1
    validation_split=0.2  # Pour diviser automatiquement les données en ensembles d'entraînement et de validation
)

# Charger les images à partir des dossiers avec le générateur flow_from_directory
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',  # Pour la classification multiclasse
    subset='training'  # Spécifier qu'il s'agit de l'ensemble d'entraînement
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Spécifier qu'il s'agit de l'ensemble de validation
)
num_classes = len(train_generator.class_indices)  # Nombre de classes dans vos données



# Définir l'architecture du modèle
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

test_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Utilisez un autre sous-ensemble si nécessaire
)


# Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Évaluation sur l'ensemble de validation
validation_loss, validation_accuracy = model.evaluate(validation_generator)
print(f"Validation accuracy: {validation_accuracy}")

# Évaluation sur l'ensemble de test
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy}")