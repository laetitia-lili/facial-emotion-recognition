import cv2
import numpy as np
import tensorflow as tf

# charge le modèle entraîné
model = tf.keras.models.load_model("../models/emotion_model.keras")

# liste des émotions dans le même ordre que pendant l'entraînement
emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# charge le détecteur de visage d'OpenCV (déjà fourni avec la bibliothèque)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ouvre la webcam (0 = webcam par défaut)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # recadre le visage détecté
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=(0, -1))  # ajoute les dimensions batch et canal

        # prédiction
        prediction = model.predict(face, verbose=0)
        emotion = emotions[np.argmax(prediction)]

        # affichage
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()