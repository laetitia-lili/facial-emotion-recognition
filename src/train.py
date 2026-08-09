import tensorflow as tf
from data_loader import get_dataset_path
from preprocessing import load_dataset, normalize_dataset
from model import build_model

# récupère les données
path = get_dataset_path()
train_ds = load_dataset(path, split="train", validation_split=0.2, subset="training")
val_ds = load_dataset(path, split="train", validation_split=0.2, subset="validation")

train_ds = normalize_dataset(train_ds)
val_ds = normalize_dataset(val_ds)

# construit et compile le modèle
model = build_model()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# entraîne le modèle
history = model.fit(train_ds, validation_data=val_ds, epochs=15)

# sauvegarde le modèle entraîné
model.save("../models/emotion_model.h5")
print("Modèle sauvegardé !")
model.save("../models/emotion_model.keras")