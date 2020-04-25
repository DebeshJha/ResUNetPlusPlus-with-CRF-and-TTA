
import os
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
from utils import create_dir

def load_data(path):
    img_path = glob(os.path.join(path, "images/*"))
    msk_path = glob(os.path.join(path, "masks/*"))

    img_path.sort()
    msk_path.sort()

    return img_path, msk_path

def colon_db(path):
    img_path = []
    msk_path = []

    for i in range(380):
        img_path.append(path + str(i+1) + ".tiff")
        msk_path.append(path + "p" + str(i+1) + ".tiff")

    img_path.sort()
    msk_path.sort()

    return img_path, msk_path

def save_data(images, masks, save_path):
    size = (256, 256)

    path = images[0].split("/")[1]
    create_dir(f"{save_path}/{path}/image")
    create_dir(f"{save_path}/{path}/mask")

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        i = cv2.imread(x, cv2.IMREAD_COLOR)
        m = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

        i = cv2.resize(i, size)
        m = cv2.resize(m, size)

        tmp_image_name = f"{idx}.jpg"
        tmp_mask_name  = f"{idx}.jpg"

        image_path = os.path.join(save_path, path, "image/", tmp_image_name)
        mask_path  = os.path.join(save_path, path, "mask/", tmp_mask_name)

        cv2.imwrite(image_path, i)
        cv2.imwrite(mask_path, m)

if __name__ == "__main__":
    save_path = "cs_data/"
    create_dir(save_path)

    paths = ["data/CVC-612"]
    for path in paths:
        x, y = load_data(path)
        save_data(x, y, save_path)

