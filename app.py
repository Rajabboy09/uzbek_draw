import os
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from io import BytesIO
from PIL import Image

# Railway RAMini tejash uchun
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

app = Flask(__name__)

# Modelni yuklash
model = tf.keras.models.load_model('quick_draw_model.h5')

# 46 ta kategoriya (Tartib trening bilan bir xil bo'lishi shart)
LABELS = [
    "Olma", "Mashina", "Uy", "Quyosh", "Daraxt", "Velosiped", "Yulduz", "Banan", "Soat", "Bulut", 
    "Eshik", "Ko'z", "Baliq", "Gul", "Gitara", "Shlyapa", "Barg", "Oy", "Tog'", "Shim", 
    "Stol", "Mushuk", "It", "Qoshiq", "Krovat", "Kitob", "Kapalak", "Tort", "Kursi", 
    "Piyola", "Quloq", "Oyoq", "Qo'l", "Muzqaymoq", "Kalit", "Burun", 
    "Nok", "Qalam", "Quyon", "Kamalak", "Ilon", "Qorbola", "Paypoq", "Soyabon", "Tarvuz", "G'ildirak"
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        
        # Tasvirni qayta ishlash
        img = Image.open(BytesIO(base64.b64decode(image_data))).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img)
        
        # Ranglarni invert qilish (AI qora fonda oq chiziqni taniydi)
        img_array = 255 - img_array
        img_array = img_array.reshape(1, 28, 28, 1) / 255.0
        
        prediction = model.predict(img_array, verbose=0)
        idx = np.argmax(prediction)
        
        return jsonify({
            'label': LABELS[idx],
            'confidence': float(np.max(prediction))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8888))
    app.run(host='0.0.0.0', port=port)