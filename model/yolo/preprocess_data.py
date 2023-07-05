# some_file.py
import sys
sys.path.insert(1, r'C:\Users\annch\OneDrive\Desktop\master\ocr')

from config import DatasetConfig, ConstantConfig
from tqdm import tqdm
from os import listdir, mkdir
from os.path import isfile, join
import shutil

train_data_path = DatasetConfig().file_path
encoder = ConstantConfig().ENCODER


def create_dirs(parent_dir):
    for i in range(1, 34):
        mkdir(f'{parent_dir}/{i}')

# create_dirs(r"C:\Users\annch\OneDrive\Desktop\master\ხელნაწერები\board")
train_path = r'C:\Users\annch\OneDrive\Desktop\master\ocr\dataset'
val_path = r'C:\Users\annch\OneDrive\Desktop\სამაგისტრო\ocr\model\yolo\dataset\val'


create_dirs(train_path)
# create_dirs(val_path)


def create_train_val_data(train_path, val_path):
    val_count = {}
    for f in tqdm(listdir(train_data_path)):
        if isfile(join(train_data_path, f)) or f == 'board':
            continue
        dir_path = join(train_data_path, f)
        for idx, imgs in enumerate(listdir(dir_path)):
            path = join(dir_path, imgs)
            key = [c for c in imgs if not 0 <= ord(c) <= 127][0]
            dir_name = str(encoder[key])
            # if val_count.get(dir_name, 0) < 50:
            #     shutil.copy2(path, val_path + '/' + dir_name + '/' + f + str(idx) + '.png')
            #     val_count[dir_name] = val_count.get(dir_name, 0) + 1
            # else:
            shutil.copy2(path, train_path + '/' + dir_name + '/' + f + str(idx) + '.png')


create_train_val_data(train_path, val_path)
