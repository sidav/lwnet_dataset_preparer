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

OVERLAP_IN_PIXELS = 0
OFFSET_FROM_BOTTOM = 80  # dies ist für die Ausnahme des Maßtab
SLICE_SIZE = 256


def slice_and_save(curr_img, folder, img_name):
    Path(folder).mkdir(parents=True, exist_ok=True)
    curynum = 0
    for fromy in range(0, len(curr_img), SLICE_SIZE):
        curxnum = 0
        for fromx in range(0, len(curr_img[0]), SLICE_SIZE):
            resulting_image = np.empty((SLICE_SIZE, SLICE_SIZE, curr_img.shape[2]))
            for y in range(fromy, fromy+SLICE_SIZE):
                for x in range(fromx, fromx+SLICE_SIZE):
                    # print(curr_img[x][y])
                    resulting_image[y-fromy][x-fromx] = curr_img[y][x]
            parcel_name = folder+"nonrand_x"+str(curxnum)+"_y"+str(curynum)+"_"+img_name
            print(parcel_name)
            cv2.imwrite(parcel_name, resulting_image)
            curxnum+=1
        curynum+=1


def do_things_with_images(image, manual, filename):
    W, H = len(image), len(image[0])
    image = cv2.resize(image, (H, W))
    manual = cv2.resize(manual, (H, W))
    slice_and_save(image, "sliced/raw/", filename)
    slice_and_save(manual, "sliced/marked/", filename)


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
