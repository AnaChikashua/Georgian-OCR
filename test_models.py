import cv2
import numpy as np
from bidict import bidict
from tensorflow import keras

ENCODER = bidict({
    'ა': 1, 'ბ': 2, 'გ': 3, 'დ': 4, 'ე': 5, 'ვ': 6,
    'ზ': 7, 'თ': 8, 'ი': 9, 'კ': 10, 'ლ': 11, 'მ': 12,
    'ნ': 13, 'ო': 14, 'პ': 15, 'ჟ': 16, 'რ': 17, 'ს': 18,
    'ტ': 19, 'უ': 20, 'ფ': 21, 'ქ': 22, 'ღ': 23, 'ყ': 24,
    'შ': 25, 'ჩ': 26, 'ც': 27, 'ძ': 28, 'წ': 29, 'ჭ': 30, 'ხ': 31, 'ჯ': 32, 'ჰ': 33
})
model_path = 'geo_model.model'
model = keras.models.load_model(model_path)


def test_model(file_path):
    img_vector = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE).astype("float32")
    img_vector = cv2.resize(img_vector, (50, 50), interpolation=cv2.INTER_AREA) / 255
    img_vector = np.expand_dims(img_vector, -1)
    pred_letter = np.argmax(model.predict(np.array([img_vector])), axis=-1)
    pred_letter = ENCODER.inverse[pred_letter[0]]
    return pred_letter


print(test_model('result/Untitled._1.png'))
