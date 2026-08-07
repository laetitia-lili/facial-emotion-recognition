import sys
sys.path.append("../src")

import matplotlib.pyplot as plt
from data_loader import get_dataset_path, count_images_per_emotion

import os
import random
from PIL import Image

# récupère le dataset et compte les images par émotion
path = get_dataset_path()
counts = count_images_per_emotion(path, split="train")

# affiche un histogramme du nombre d'images par émotion
plt.bar(counts.keys(), counts.values())
plt.title("Nombre d'images par émotion (train)")
plt.xlabel("Émotion")
plt.ylabel("Nombre d'images")
plt.show()

# affiche 7 images aléatoires, une par émotion
emotions = list(counts.keys())
fig, axes = plt.subplots(1, len(emotions), figsize=(15, 3))

for i, emotion in enumerate(emotions):
    emotion_folder = os.path.join(path, "train", emotion)
    images = os.listdir(emotion_folder)
    random_image = random.choice(images)

    img_path = os.path.join(emotion_folder, random_image)
    img = Image.open(img_path)

    axes[i].imshow(img, cmap="gray")
    axes[i].set_title(emotion)
    axes[i].axis("off")

plt.tight_layout()
plt.show()