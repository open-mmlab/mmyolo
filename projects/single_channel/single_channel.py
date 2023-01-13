import imghdr
from typing import List

from PIL import Image
import os

path = 'projects/single_channel/data/balloon/train'  # Read the path of the image folder
save_path = 'projects/single_channel/data/balloon/train'  # Save image folder path
file_list: List[str] = os.listdir(path)

# Grayscale conversion of each image
for file in file_list:
    if imghdr.what(path + '/' + file) != 'jpeg':
        continue
    o_img = Image.open(path + '/' + file)
    L_img = o_img.convert('L')
    L_img.save(save_path + '/' + file)

path = 'projects/single_channel/data/balloon/val'  # Read the path of the image folder
save_path = 'projects/single_channel/data/balloon/val'  # Save image folder path
file_list = os.listdir(path)

# Grayscale conversion of each image
for file in file_list:
    if imghdr.what(path + '/' + file) != 'jpeg':
        continue
    o_img = Image.open(path + '/' + file)
    L_img = o_img.convert('L')
    L_img.save(save_path + '/' + file)
