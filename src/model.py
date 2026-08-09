import tensorflow as tf
from tensorflow.keras import layers, models

# construit l'architecture du CNN pour la classification d'émotions
def build_model(input_shape=(48, 48, 1), num_classes=7):
    model = models.Sequential([

        layers.Input(shape=input_shape),

        # détecte des motifs simples (contours, textures)
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # détecte des motifs plus complexes
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # motifs encore plus abstraits
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # aplatit les données 2D en une seule ligne
        layers.Flatten(),

        # couche dense avec dropout pour éviter l'overfitting
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),

        # couche de sortie: une probabilité par émotion
        layers.Dense(num_classes, activation="softmax")
    ])

    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()