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
source_dir = 'result/images' # 'new_test_x'
preds_dir = 'result/predictions' # 'new_test_y'
mask_dir = 'result/manual' # 'new_test_y'

def do_things_with_images(image, pred, mask, filename):
    if len(np.shape(image)) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if len(np.shape(mask)) != 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # merged = cv2.hconcat(image, manual)
    merged = np.concatenate((image, pred), axis=1)
    merged = np.concatenate((merged, mask), axis=1)
    cv2.imwrite("result/concat/"+filename, merged)


Path("result/concat").mkdir(parents=True, exist_ok=True)
for filename in os.listdir(source_dir):
    print("\nWorking on " + filename + "...")
    source_img = np.array(Image.open(source_dir+'/'+filename))
    test_img = np.array(Image.open(preds_dir + '/' + filename))
    mask_img = np.array(Image.open(mask_dir + '/' + filename))
    do_things_with_images(source_img, test_img, mask_img, filename)
