import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Klassen definieren
CLASS_NAMES = ["Shirt", "Jacke", "Schuhe", "Trinkflasche", "Sonstiges"]

# Vortrainiertes Modell laden (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# Eigenen Klassifikationskopf hinzufügen
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")
])

# ⚠️ Falls du ein eigenes trainiertes Modell hast:
# model.load_weights("fundsachen_model.h5")

st.title("🔍 Fundsachen-Erkennung")

uploaded_file = st.file_uploader("Lade ein Bild hoch", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # Bild vorbereiten
    image_resized = image.resize((224, 224))
    img_array = np.array(image_resized)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Vorhersage
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.subheader(f"📌 Ergebnis: {predicted_class}")
    st.write(f"🔎 Sicherheit: {confidence:.2f}%")
