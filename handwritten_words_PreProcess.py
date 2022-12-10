import numpy as np
import cv2
import pickle

import os
from data_generator import MedIMG
import config

DATA_PATH = config.RAW_HANDWRITTEN_WORDS_PATH
OUTPUT_PATH = config.HANDWRITTEN_WORDS_PATH

if not os.path.exists(f"{OUTPUT_PATH}"):  # creating out put directory
    os.makedirs(f"{OUTPUT_PATH}")

OUTPUT_LIST_ID = []
pic_num = 1
pic_name = f"TRAIN_{str(pic_num).zfill(5)}.jpg"
for counter in range(1000):

    pic_name = f"TRAIN_{str(pic_num).zfill(5)}.jpg"
    pic_num += 1
    if not os.path.isfile(DATA_PATH + pic_name):
        continue

    top_crop_value = 0
    bottom_crop_value = 0
    left_crop_value = 0
    right_crop_value = 0

    img = cv2.imread(DATA_PATH + pic_name, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(pic_name)
        raise FileNotFoundError

    BLACKNESS_THRESHOLD = 255 - MedIMG.WHITENESS_THRESHOLD
    for c, row in enumerate(img):
        if np.amin(row) < BLACKNESS_THRESHOLD:

            if top_crop_value == 0:
                top_crop_value = c
            elif c > bottom_crop_value:
                bottom_crop_value = c

    for c, row in enumerate(img.T):
        if np.amin(row) < BLACKNESS_THRESHOLD:

            if left_crop_value == 0:
                left_crop_value = c
            elif c > right_crop_value:
                right_crop_value = c

    top_crop_value -= 3
    bottom_crop_value += 3
    left_crop_value -= 3
    right_crop_value += 3
    crop_img = img[top_crop_value:bottom_crop_value, left_crop_value:right_crop_value]

    # saving processed image
    writeStatus = cv2.imwrite(OUTPUT_PATH + pic_name, crop_img)
    if writeStatus is None:
        print(pic_name)
        raise OSError
    OUTPUT_LIST_ID.append(pic_name)

with open(config.HANDWRITTEN_WORDS_IDS_PATH, "wb") as fp:  # Pickling
    pickle.dump(OUTPUT_LIST_ID, fp)

