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

# bloc test 
if __name__ == "__main__":
    from data_loader import get_dataset_path

    path = get_dataset_path()
    train_ds = load_dataset(path, split="train")

    print("Classes trouvées :", train_ds.class_names)
    