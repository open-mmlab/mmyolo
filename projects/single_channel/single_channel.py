import imghdr
import os
from typing import List

from PIL import Image

path = 'projects/single_channel/data/balloon/train'
save_path = 'projects/single_channel/data/balloon/train'
file_list: List[str] = os.listdir(path)

# Grayscale conversion of each image
for file in file_list:
    if imghdr.what(path + '/' + file) != 'jpeg':
        continue
    o_img = Image.open(path + '/' + file)
    L_img = o_img.convert('L')
    L_img.save(save_path + '/' + file)

path = 'projects/single_channel/data/balloon/val'
save_path = 'projects/single_channel/data/balloon/val'
file_list = os.listdir(path)

# Grayscale conversion of each image
for file in file_list:
    if imghdr.what(path + '/' + file) != 'jpeg':
        continue
    o_img = Image.open(path + '/' + file)
    L_img = o_img.convert('L')
    L_img.save(save_path + '/' + file)
