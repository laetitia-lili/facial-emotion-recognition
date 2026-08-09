import tensorflow as tf
import os

# charge les images d'un dossier (train ou test) sous forme de dataset TensorFlow prêt à l'emploi
def load_dataset(dataset_path, split="train", image_size=(48, 48), batch_size=32, validation_split=None, subset=None):
    split_path = os.path.join(dataset_path, split)

    dataset = tf.keras.utils.image_dataset_from_directory(
        split_path,
        labels="inferred",
        label_mode="categorical",
        color_mode="grayscale",
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        subset=subset,
        seed=123
    )

    return dataset

# normalise les pixels d'un dataset (0-255 -> 0-1)
def normalize_dataset(dataset):
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
    return normalized_dataset

# bloc test 
if __name__ == "__main__":
    from data_loader import get_dataset_path
    import numpy as np

    path = get_dataset_path()

    train_ds = load_dataset(path, split="train", validation_split=0.2, subset="training")
    val_ds = load_dataset(path, split="train", validation_split=0.2, subset="validation")

    print("Classes trouvées :", train_ds.class_names)

    train_ds = normalize_dataset(train_ds)

    for images, labels in train_ds.take(1):
        print("Valeur min des pixels :", np.min(images.numpy()))
        print("Valeur max des pixels :", np.max(images.numpy()))
    