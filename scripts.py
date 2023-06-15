## For joining all images in one dir
import os
from os import listdir
from datasets import load_dataset
def join_images():
    dir_path = r'C:\Users\annch\OneDrive\Desktop\სამაგისტრო\handwriting_dataset'
    for sub_dir in listdir(dir_path):
        sub_dir = dir_path + '\\' + sub_dir
        source = listdir(sub_dir)
        for f in source:
            src_path = os.path.join(sub_dir, f)
            dst_path = os.path.join(dir_path, f)
            os.rename(src_path, dst_path)
        os.rmdir(sub_dir)
        print(sub_dir)
def load_huggingface_dataset():
    return load_dataset('AnaChikashua/handwriting', split='train')