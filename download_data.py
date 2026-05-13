import urllib.request
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


categories = [
    'apple', 'car', 'house', 'sun', 'tree', 'bicycle', 'star', 'banana', 'alarm clock', 'cloud',
    'door', 'eye', 'fish', 'flower', 'guitar', 'hat', 'leaf', 'moon', 'mountain', 'pants',
    'table', 'cat', 'dog', 'spoon', 'bed', 'book', 'butterfly', 'cake', 'chair',
    'cup', 'ear', 'foot', 'hand', 'ice cream', 'key', 'nose',
    'pear', 'pencil', 'rabbit', 'rainbow', 'snake', 'snowman', 'sock', 'umbrella', 'watermelon', 'wheel'
]

base_url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
os.makedirs('data', exist_ok=True)

headers = {'User-Agent': 'Mozilla/5.0'}

for cat in categories:
    file_name = f"{cat}.npy"
    if not os.path.exists(f"data/{file_name}"):
        print(f"Yuklanmoqda: {file_name}...")
        url = (base_url + file_name).replace(' ', '%20')
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(f"data/{file_name}", 'wb') as f:
            f.write(response.read())