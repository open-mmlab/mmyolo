import imghdr
import os
import argparse
from typing import List

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='data_path')
    parser.add_argument(
        "--path",
        type=str,
        default="projects/single_channel/data/balloon",
        help="Original dataset path")
    return parser.parse_args()


def main():
    args = parse_args()

    path = args.path + '/train/'
    save_path = path
    file_list: List[str] = os.listdir(path)
    # Grayscale conversion of each imager
    for file in file_list:
        if imghdr.what(path + '/' + file) != 'jpeg':
            continue
        o_img = Image.open(path + '/' + file)
        L_img = o_img.convert('L')
        L_img.save(save_path + '/' + file)
        args = parse_args()

    path = args.path + '/val/'
    save_path = path
    file_list: List[str] = os.listdir(path)
    # Grayscale conversion of each imager
    for file in file_list:
        if imghdr.what(path + '/' + file) != 'jpeg':
            continue
        o_img = Image.open(path + '/' + file)
        L_img = o_img.convert('L')
        L_img.save(save_path + '/' + file)


if (__name__ == "__main__"):
    main()
