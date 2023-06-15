from sklearn.utils import shuffle
import numpy as np
class Preprocessing:
    def get_label_data(self, img_data):
        labels = []
        imgs = np.array([val for ob in img_data.values() for val in ob])
        for key, value in img_data.items():
            labels.extend([key]* len(value))
        labels = np.array(labels)
        labels, imgs = shuffle(labels, imgs)
        return labels, imgs
    
    def train_test_split(self, img_data, split=.75):
        labels, imgs = self.get_label_data(img_data)
        labels_train = labels[:int(len(labels) * split)]
        labels_test = labels[int(len(labels) * split):]
        imgs_train = imgs[:int(len(imgs) * split)]
        imgs_test = imgs[int(len(imgs) * split):]
        return labels_train, labels_test, imgs_train, imgs_test