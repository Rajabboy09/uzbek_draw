from flask import Flask, request, jsonify, render_template
import base64
import numpy as np
from PIL import Image
import io
import tensorflow as tf

app = Flask(__name__)

# 1. Tayyor modelni yuklaymiz
model = tf.keras.models.load_model('quick_draw_model.h5')

# 2. Kategoriyalar (Training skriptidagi tartib bilan bir xil bo'lishi shart)
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
        data = request.json['image']
        image_data = base64.b64decode(data.split(',')[1])
        img = Image.open(io.BytesIO(image_data))

        # AGAR RASM RGBA (shaffof) BO'LSA, UNI OQ FONGA QO'YAMIZ
        if img.mode == 'RGBA':
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3]) # 3 - alpha kanali
            img = background.convert('L')
        else:
            img = img.convert('L')

        img = img.resize((28, 28))
        img_array = np.array(img)
        
        # Google Dataset qora fonda oq chiziq (shuning uchun teskari qilamiz)
        img_array = 255 - img_array 
        
        # DEBUG: AI nima ko'rayotganini rasm qilib saqlaymiz
        debug_img = Image.fromarray(img_array.astype('uint8'))
        debug_img.save("ai_debug.png")

        img_array = img_array.reshape(1, 28, 28, 1).astype('float32') / 255.0
        
        prediction = model.predict(img_array)
        result_index = np.argmax(prediction)
        
        return jsonify({
            "label": LABELS[result_index],
            "confidence": f"{float(prediction[0][result_index]) * 100:.1f}%"
        })
    except Exception as e:
        print(f"Xato: {e}")
        return jsonify({"error": str(e)}), 400
    
if __name__ == '__main__':
    app.run(debug=True)