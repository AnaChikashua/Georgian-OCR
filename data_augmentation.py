from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import DatasetConfig, ConstantConfig
import random
import albumentations as alb

# ---------------------- https://towardsdatascience.com/effective-data-augmentation-for-ocr-8013080aa9fa


class DataAugmentation:
    train_data_path = DatasetConfig().file_path
    encoder = ConstantConfig().ENCODER

    def __init__(self) -> None:
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def get_data(self) -> dict:
        img_data = {}
        for f in tqdm(listdir(self.train_data_path)):
            if isfile(join(self.train_data_path, f)):
                continue
            dir_path = join(self.train_data_path, f)
            for imgs in listdir(dir_path):
                path = join(dir_path, imgs)
                img_vector = cv2.imdecode(np.fromfile(
                    path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE).astype("float32")
                img_vector = cv2.resize(
                    img_vector, (50, 50), interpolation=cv2.INTER_AREA) / 255
                img_vector = np.expand_dims(img_vector, -1)
                key = [c for c in imgs if not 0 <= ord(c) <= 127][0]
                img_data.setdefault(self.encoder[key], []).append(img_vector)
        return img_data

    def morphological_alteration(self, img):
        """
        Used to make the text lines appear to be written with a finer or thicker pen.
        """
        # morphological alterations
        if random.randint(1, 5) == 1:
            # dilation because the image is not inverted
            img = cv2.erode(img, self.kernel, iterations=random.randint(1, 1))
        if random.randint(1, 6) == 1:
            # erosion because the image is not inverted
            img = cv2.dilate(img, self.kernel, iterations=random.randint(1, 1))
        return img

    @staticmethod
    def transform_image():
        transform = alb.Compose([
            # alb.OneOf([
            # add black pixels noise
            # alb.OneOf([
            #     alb.RandomRain(slant_lower = 0, slant_upper=0, brightness_coefficient=1.0, drop_length=2, drop_width=2, drop_color = (0, 0, 0), blur_value=1, rain_type = 'drizzle', p=0.05),
            #     alb.RandomShadow(p=1),
            #     alb.PixelDropout(p=1),
            # ], p=0.9),

            # add white pixels noise
            # alb.OneOf([
            #     # alb.PixelDropout(dropout_prob=0.9, drop_value=255, p=1),
            #     alb.RandomRain(brightness_coefficient=1.0, drop_length=2, drop_width=2, drop_color=(
            #         255, 255, 255), blur_value=1, rain_type = None, p=0.5),], p=0.9),
            # ], p=1),

            # transformations
            alb.OneOf([
                alb.ShiftScaleRotate(shift_limit=0, scale_limit=0.25, rotate_limit=2,
                                     value=(255, 255, 255), p=1),
                alb.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=8,
                                     value=(255, 255, 255), p=1),
                alb.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.15, rotate_limit=11,
                                     value=(255, 255, 255), p=1),
                alb.Affine(shear=random.randint(-5, 5),
                           mode=cv2.BORDER_CONSTANT, cval=(255, 255, 255), p=1)
            ], p=0.5), alb.Blur(blur_limit=5, p=0.25),
        ])
        return transform

    def augment_img(self, img):
        # only augment 3/4th the images
        if random.randint(1, 4) > 3:
            return []
        # img = np.asarray(img)     #convert to numpy for opencv
        img = self.morphological_alteration(img)
        transform = self.transform_image()
        try:
            img = transform(image=img)['image']
        except:
            return []
        # image = Image.fromarray(img)
        return img

    def data_visualization(self, img):
        plt.figure()
        plt.imshow(img)
        plt.grid(False)
        plt.show()

    def process_data(self):
        data = self.get_data()
        for char in tqdm(data):
            for img in data[char]:
                augment_img = self.augment_img(img)
                if not len(augment_img):
                    continue
                # self.data_visualization(img)
                # self.data_visualization(augment_img)
                if augment_img.shape != (50, 50, 1):
                    augment_img = np.expand_dims(augment_img, -1)
                data[char].append(augment_img)
        return data


if __name__ == "__main__":
    DataAugmentation().process_data()
