from flask import Flask, request, jsonify
import tensorflow as tf
import cv2
import numpy as np
import os
import base64
import sys
sys.stdout.reconfigure(encoding='utf-8')



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(BASE_DIR, "static", "json", "network.json")
hdf5 = os.path.join(BASE_DIR, "static", "hdf5", "weights.hdf5")

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"Home": "Vc está na Home"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        img_data_64 = data.get("image")
        
        if not img_data_64:
            return jsonify({"error": "Nenhuma imagem fornecida"}), 400
        
        img_bytes = base64.b64decode(img_data_64)
        imagem_teste = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        if imagem_teste is None:
            return jsonify({"error": "Erro ao decodificar a imagem"}), 400

        
        image_padronizada = cv2.resize(imagem_teste, (64, 64)) / 255.0 # Redimensionando
        image_padronizada = image_padronizada.reshape(-1, 64, 64, 3) # Normalizando a imagem

        
        with open(json_path, 'r', encoding='utf-8') as json_file: # Carregar modelo
            json_saved_model = json_file.read()
            network_loaded = tf.keras.models.model_from_json(json_saved_model)
            network_loaded.load_weights(hdf5)
            network_loaded.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    
        previsao = network_loaded.predict(image_padronizada)
        result = 'Parasitized' if np.argmax(previsao) == 0 else 'Uninfected'

        return jsonify({'predict': result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
