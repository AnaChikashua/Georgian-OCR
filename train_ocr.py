import cv2
from os import listdir
from os.path import isfile, join
from os import walk
import numpy as np
train_data_path = r'C:\Users\annch\OneDrive\Desktop\სამაგისტრო\ხელნაწერები'
img_data = {}
im = cv2.imread(r"C:\Users\annch\OneDrive\Desktop\სამაგისტრო\ხელნაწერები\GHSF_0\GHSF_0__ა_m_1.png", cv2.IMREAD_UNCHANGED)
for f in listdir(train_data_path):
    if isfile(join(train_data_path, f)):
        continue
    dir_path = join(train_data_path, f)
    for imgs in listdir(dir_path):
        path = join(dir_path, imgs)
        img_vector = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED).resize(1, 50, 50, refcheck=False)
        img_data.setdefault(imgs[8],[]).append(img_vector)
    