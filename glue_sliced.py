import shutil
from copy import deepcopy

from transform import elastic_transform
import os
import numpy as np
import re
import cv2

from PIL import Image
from pathlib import Path

toslice_folder = "toslice/"
source_dir = 'result/images' # 'new_test_x'
preds_dir = 'result/predictions' # 'new_test_y'
mask_dir = 'result/manual' # 'new_test_y'


class Parcel:
    def __init__(self, x, y, img):
        self.x = int(x)
        self.y = int(y)
        self.img = img


class ParcelPack:
    def __init__(self, name):
        self.parcels = []
        self.name = name
        self.arr = []

    def add(self, prc):
        self.parcels.append(prc)

    def form_array_of_parcels(self):
        highest_x = 0
        highest_y = 0
        for p in self.parcels:
            if p.x > highest_x:
                highest_x = p.x
            if p.y > highest_y:
                highest_y = p.y
        arr = [[None] * (highest_y+1) for _ in range(highest_x + 1)]
        for p in self.parcels:
            # print(p.x, p.y, len(arr))
            arr[p.x][p.y] = p
        self.arr = arr

    def glue_together_and_save(self):
        self.form_array_of_parcels()
        glued_rows = []
        # first, glue row one parcel by one parcel
        for row in range(len(self.arr)):
            curr_row_img = None
            for col in range(len(self.arr[0])):
                curp = self.arr[row][col]
                image = curp.img
                if len(np.shape(image)) != 2:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if curr_row_img is None:
                    curr_row_img = image
                else:
                    curr_row_img = np.concatenate((curr_row_img, image), axis=0)
            glued_rows.append(curr_row_img)
        merged = None
        for row in range(len(glued_rows)):
            rowimg = glued_rows[row]
            if merged is None:
                merged = rowimg
            else:
                merged = np.concatenate((merged, rowimg), axis=1)
        cv2.imwrite("result/glued/glued_"+self.name+".png", merged)


Path("result/glued").mkdir(parents=True, exist_ok=True)
parcel_packs = {}
for filename in os.listdir(preds_dir):
    print("\nWorking on " + filename + "...")
    test_img = np.array(Image.open(preds_dir + '/' + filename))
    x = re.search("(?<=\_x).+(?=\_y)", filename).group()
    y = re.search("(?<=\_y).+(?=\_)", filename).group()
    name = re.search("(?<=\_)[^_]+(?=\.)", filename).group()
    if "transformed" in filename:
        name = "transformed_" + re.search("(?<=transformed\_)[^_]+(?=\_)", filename).group() + "_" + name
    if name not in parcel_packs.keys():
        parcel_packs[name] = ParcelPack(name)
        print("Creating parcel pack " + name)
    print("Adding parcel for pack" + name + " with x: " + x + "; y: " + y)
    prc = Parcel(x, y, test_img)
    parcel_packs[name].add(prc)

for pack in parcel_packs.values():
    print("GLUEING PARCEL PACK ", pack.name)
    pack.glue_together_and_save()
