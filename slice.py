import shutil
from copy import deepcopy

from transform import elastic_transform
import os
import numpy as np
import sys
import cv2

from PIL import Image
from pathlib import Path

toslice_folder = "toslice/"
source_dir = 'toslice/raw' # 'new_test_x'
handmade_dir = 'toslice/marked' # 'new_test_y'
SLICES_PER_IMAGE = 10
OFFSET_FROM_BOTTOM = 80  # dies ist für die Ausnahme des Maßtab
SLICE_SIZE = 256


def slice_and_save(curr_img, folder, img_name, randomState):
    Path(folder).mkdir(parents=True, exist_ok=True)
    for num in range(SLICES_PER_IMAGE):
        fromx = randomState.randint(0, len(curr_img)-SLICE_SIZE)
        fromy = randomState.randint(0, len(curr_img[0]) - SLICE_SIZE - OFFSET_FROM_BOTTOM)
        resulting_image = np.empty((SLICE_SIZE, SLICE_SIZE, curr_img.shape[2]))
        for x in range(fromx, fromx+SLICE_SIZE):
            for y in range(fromy, fromy+SLICE_SIZE):
                # print(curr_img[x][y])
                resulting_image[x-fromx][y-fromy] = curr_img[x][y]
        print(folder+str(num)+img_name)
        cv2.imwrite(folder+str(num)+img_name, resulting_image)


def do_things_with_images(image, manual, filename):
    H, W = len(image), len(image[0])
    image = cv2.resize(image, (H, W))
    manual = cv2.resize(manual, (H, W))
    random_state_image = np.random.RandomState(None)
    random_state_manual = deepcopy(random_state_image)
    slice_and_save(image, "sliced/raw/", filename, random_state_image)
    slice_and_save(manual, "sliced/marked/", filename, random_state_manual)


try:
    shutil.rmtree("sliced")
except:
    print("Nothing to delete.")
Path("sliced").mkdir(parents=True, exist_ok=True)
for filename in os.listdir(source_dir):
    print("\nWorking on " + filename + "...")
    source_img = np.array(Image.open(source_dir+'/'+filename))
    test_img = np.array(Image.open(handmade_dir+'/'+filename))
    do_things_with_images(source_img, test_img, filename)
