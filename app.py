# app.py - API Flask compatible con modelo_mnist.h5
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# === CARGAR MODELO ===
print("Cargando modelo 'modelo_mnist.h5'...")
model = load_model("modelo_mnist.h5", compile=False)  # ← Evita warnings
print("Modelo cargado exitosamente.")

# === CARPETA DE SUBIDAS ===
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Guardar imagen
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocesar
        img = Image.open(filepath).convert('L')  # Grises
        img = img.resize((28, 28))
        img_array = np.array(img).astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predecir
        prediction = model.predict(img_array, verbose=0)
        digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return jsonify({
            'prediccion': digit,
            'confianza': round(confidence, 4),
            'imagen': file.filename
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)