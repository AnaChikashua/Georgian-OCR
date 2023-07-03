import numpy as np
from ultralytics import YOLO
from bidict import bidict
ENCODER = bidict({
        'ა': 1, 'ბ': 2, 'გ': 3, 'დ': 4, 'ე': 5, 'ვ': 6,
        'ზ': 7, 'თ': 8, 'ი': 9, 'კ': 10, 'ლ': 11, 'მ': 12,
        'ნ': 13, 'ო': 14, 'პ': 15, 'ჟ': 16, 'რ': 17, 'ს': 18,
        'ტ': 19, 'უ': 20, 'ფ': 21, 'ქ': 22, 'ღ': 23, 'ყ': 24,
        'შ': 25, 'ჩ': 26, 'ც': 27, 'ძ': 28, 'წ': 29, 'ჭ': 30, 'ხ': 31, 'ჯ': 32, 'ჰ': 33
    })# Load a model
model = YOLO(r'C:\Users\annch\OneDrive\Desktop\master\ocr\runs\classify\train6\weights\best.pt')  # load a custom model

# Predict with the model
results = model(r'result\word0_1-0_5.png')
names_dict = results[0].names
probs = results[0].probs.tolist()

print(ENCODER.inverse[int(names_dict[np.argmax(probs)])])
