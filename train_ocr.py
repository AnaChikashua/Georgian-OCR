import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
from bidict import bidict
from tqdm import tqdm

from sklearn.utils import shuffle
from tensorflow import keras

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from custom_models import custom_model, old_model
from data_preprocessing import Preprocessing
from data_augmentation import DataAugmentation
from config import ConstantConfig

class TrainModel:
    encoder = ConstantConfig().ENCODER
    batch_size = 16
    epochs = 20
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2)

    def __init__(self) -> None:
        self.preprocessing = Preprocessing()
        self.augmentation = DataAugmentation()
        img_data = self.augmentation.process_data()
        self.labels_train, self.labels_test, self.imgs_train, self.imgs_test = self.preprocessing.train_test_split(img_data)
        self.model = old_model()
        # model = custom_model()

    def train_model(self):
        self.model.fit(self.imgs_train, self.labels_train, batch_size=self.batch_size, epochs=self.epochs,
                       validation_data=(self.imgs_test, self.labels_test), callbacks=[self.early_stopping])
        
    def test_model(self, test_data=None):
        if test_data:
            return np.argmax(self.model.predict(test_data), axis=1)
        return np.argmax(self.model.predict(self.imgs_test), axis=1)
    
    def confusion_matrix(self):
        cm = confusion_matrix(self.labels_test, self.labels_pred,
                              labels=list(self.encoder.inverse.keys()))
        plt.figure(figsize=(12, 12))
        sns.heatmap(cm, annot=True, cbar=False, cmap='Blues', xticklabels=list(
            self.encoder.keys()), yticklabels=list(self.encoder.keys()))
        plt.show()

    def save_model(self, model_name='geo_model.model'):
        self.model.save(model_name, save_format='h5')
