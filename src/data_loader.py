import kagglehub
import os

# télécharge (ou récupère depuis le cache) le dataset FER2013 et retourne le chemin vers train/ et test/
def get_dataset_path():
    path = kagglehub.dataset_download("msambare/fer2013")
    return path

# compte le nombre d'images dans chaque dossier d'émotion (split = "train" ou "test")
def count_images_per_emotion(dataset_path, split="train"):
    split_path = os.path.join(dataset_path, split)
    emotions = os.listdir(split_path)
    counts = {}
    for emotion in emotions:
        emotion_path = os.path.join(split_path, emotion)
        nb_images = len(os.listdir(emotion_path))
        counts[emotion] = nb_images
    return counts

if __name__ == "__main__":
    path = get_dataset_path()
    print("Dataset disponible ici :", path)

    counts_train = count_images_per_emotion(path, split="train")
    print("Nombre d'images par émotion (train) :", counts_train)