import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils import shuffle

def load_data():
    # download_data dagi ro'yxat bilan bir xil bo'lishi shart!
    cats = [
   'apple', 'car', 'house', 'sun', 'tree', 'bicycle', 'star', 'banana', 'alarm clock', 'cloud',
    'door', 'eye', 'fish', 'flower', 'guitar', 'hat', 'leaf', 'moon', 'mountain', 'pants',
    'table', 'cat', 'dog', 'spoon', 'bed', 'book', 'butterfly', 'cake', 'chair',
    'cup', 'ear', 'foot', 'hand', 'ice cream', 'key', 'nose',
    'pear', 'pencil', 'rabbit', 'rainbow', 'snake', 'snowman', 'sock', 'umbrella', 'watermelon', 'wheel'
    ]
    
    x, y = [], []
    for i, cat in enumerate(cats):
        data = np.load(f'data/{cat}.npy')[:5000] # Har biridan 5000 tadan rasm
        x.append(data)
        y.append(np.full(data.shape[0], i))
    
    x = np.concatenate(x).reshape(-1, 28, 28, 1) / 255.0
    y = np.concatenate(y)
    return shuffle(x, y, random_state=42)

x_train, y_train = load_data()

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'), # Qo'shimcha qatlam
    layers.Flatten(),
    layers.Dense(512, activation='relu'), 
    layers.Dense(50, activation='softmax') 
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5-8 epoch o'qitish tavsiya etiladi
model.fit(x_train, y_train, epochs=7, batch_size=32)
model.save('quick_draw_model.h5')