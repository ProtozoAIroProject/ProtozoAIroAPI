from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import base64
import io
import sys
sys.stdout.reconfigure(encoding='utf-8')


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(BASE_DIR, "static", "json", "network.json")
hdf5 = os.path.join(BASE_DIR, "static", "hdf5", "weights.hdf5")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        img_data_64 = data.get("image")
        
        if not img_data_64:
            return jsonify({"error": "Nenhuma imagem fornecida"}), 400
        
        # Decodificando a imagem base64
        img_bytes = base64.b64decode(img_data_64)
        
        # Usando PIL para abrir a imagem
        img = Image.open(io.BytesIO(img_bytes))

        # Convertendo a imagem para o formato RGB e redimensionando
        img = img.convert("RGB")
        img = img.resize((64, 64))

        # Convertendo a imagem para um array NumPy e normalizando
        image_array = np.array(img) / 255.0
        image_array = image_array.reshape(-1, 64, 64, 3)  # Ajustando as dimensões

        # Carregando o modelo
        with open(json_path, 'r', encoding='utf-8') as json_file:
            json_saved_model = json_file.read()
            network_loaded = tf.keras.models.model_from_json(json_saved_model)
            network_loaded.load_weights(hdf5)
            network_loaded.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Fazendo a previsão
        previsao = network_loaded.predict(image_array)
        result = 'Parasitized' if np.argmax(previsao) == 0 else 'Uninfected'

        return jsonify({'predict': result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
