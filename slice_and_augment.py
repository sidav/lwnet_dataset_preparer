import shutil
from copy import deepcopy

from transform import elastic_transform
import os
import numpy as np
import sys
import cv2
from PIL import Image
from pathlib import Path

parent_result_folder = "result/"
source_dir = 'raw' # 'new_test_x'
handmade_dir = 'marked' # 'new_test_y'
TRANSFORMATIONS_PER_IMAGE = 2

USE_INITIAL_SIZE = True
H, W = 2048, 430  # hat keinen Effekt wenn USE_INITIAL_SIZE "True" ist.


def transform_and_save_both(curr_img, folder, transform, randomState):
    from pathlib import Path
    Path(parent_result_folder + folder).mkdir(parents=True, exist_ok=True)

    if len(np.shape(curr_img)) != 2:
        curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    curr_img = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2BGR)
    # print(np.shape(curr_img))
    # curr_img = np.expand_dims(curr_img, axis=-1)
    cv2.imwrite(parent_result_folder + folder + filename, curr_img)

    print("TEST RANDOM FOR THIS: ", randomState.random())
    if transform:
        curr_add_number = 0
        transformed = elastic_transform(curr_img, curr_img.shape[1] * 0.2, curr_img.shape[1] * 0.008,
                                        curr_img.shape[1] * 0.008, randomState)

        fname = ""
        while curr_add_number == 0 or os.path.isfile(fname):
            curr_add_number += 1
            fname = parent_result_folder + folder + "transformed_" + str(curr_add_number) + '_' + filename

        cv2.imwrite(fname, transformed)


def do_things_with_images(image, manual):
    global csv
    if USE_INITIAL_SIZE:
        H, W = image.shape[0], image.shape[1]
    image = cv2.resize(image, (H, W))
    manual = cv2.resize(manual, (H, W))
    manual = cv2.bitwise_not(manual)
    for _ in range(TRANSFORMATIONS_PER_IMAGE):
        random_state_image = np.random.RandomState(None)
        random_state_manual = deepcopy(random_state_image)

        transform_and_save_both(image, "images/", True, random_state_image)
        transform_and_save_both(manual, "manual/", True, random_state_manual)
        mask = np.full((W, H, 3), 255, np.uint8)
        # print(np.shape(mask))
        transform_and_save_both(mask, "masks/", True, random_state_image)


def create_csv():
    TEST_PERCENT = 20
    VAL_PERCENT = 20
    header = "im_paths,gt_paths,mask_paths\n"
    csv_all = header
    csv_train = header
    csv_test = header
    csv_val = header
    csv_path = "data/TRY/"
    total_files_num = len(os.listdir(parent_result_folder + "images/"))
    curr_file_num = 0
    for filename in os.listdir(parent_result_folder + "images/"):
        curr_file_num += 1
        csv_all += csv_path + "images/" + filename + ","
        csv_all += csv_path + "manual/" + filename + ","
        csv_all += csv_path + "masks/" + filename + "\n"
        if curr_file_num*100/total_files_num > TEST_PERCENT:
            csv_train += csv_path + "images/" + filename + ","
            csv_train += csv_path + "manual/" + filename + ","
            csv_train += csv_path + "masks/" + filename + "\n"
        else:
            csv_test += csv_path + "images/" + filename + ","
            csv_test += csv_path + "manual/" + filename + ","
            csv_test += csv_path + "masks/" + filename + "\n"

        if curr_file_num*100/total_files_num <= VAL_PERCENT:
            csv_val += csv_path + "images/" + filename + ","
            csv_val += csv_path + "manual/" + filename + ","
            csv_val += csv_path + "masks/" + filename + "\n"

    with open(parent_result_folder + "all.csv", "w") as text_file:
        text_file.write(csv_all)
    with open(parent_result_folder + "train.csv", "w") as text_file:
        text_file.write(csv_train)
    with open(parent_result_folder + "test.csv", "w") as text_file:
        text_file.write(csv_test)
    with open(parent_result_folder + "val.csv", "w") as text_file:
        text_file.write(csv_val)


try:
    shutil.rmtree("result")
except:
    print("Nothing to delete.")
Path("result").mkdir(parents=True, exist_ok=True)
for filename in os.listdir(source_dir):
    print("\nWorking on " + filename + "...")
    source_img = np.array(Image.open(source_dir+'/'+filename))
    test_img = np.array(Image.open(handmade_dir+'/'+filename))
    do_things_with_images(source_img, test_img)

create_csv()

    # im = cv2.imread(path + 'test_x/' + str(a[i]) + '.jpg')

    # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # im = clahe.apply(im)
    # # im = np.expand_dims(im,axis=-1)
    # im_mask = np.array(Image.open(path + 'test_y/' + str(a[i]) + '.jpg'))
    #
    # im_mask = cv2.cvtColor(im_mask, cv2.COLOR_BGR2GRAY)
    # im_mask = cv2.resize(im_mask, (H, W))
    # ret, im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #
    # print(j, im_mask.shape)
    # # draw_grid(im, 50)
    # im_merge = np.concatenate((im[..., None], im_mask[..., None]), axis=2)
    # # print(im_merge.shape)
    # im = elastic_transform(im_merge, im_merge.shape[1] * 0.2, im_merge.shape[1] * 0.008, im_merge.shape[1] * 0.008)
    # im_t = im[..., 0]
    # # im_t = image.img_to_array(im_t)
    # im_mask_t = im[..., 1]
    # im_mask_t = image.img_to_array(im_mask_t)
    # ret, im_mask = cv2.threshold(im_mask, 1, 255, cv2.THRESH_BINARY_INV)
    # im_t = image.img_to_array(im_t / 255)
    # im_mask = image.img_to_array(im_mask / 255)
    # k1 = Im.get_slices_batch(im_t)
    # k2 = Im.get_slices_batch(im_mask_t / 255)
    # train_x.extend([image.img_to_array(k) for k in k1['images']])
    # train_y.extend(
    #     [(image.img_to_array(cv2.threshold(image.img_to_array(k), 0, 255, cv2.THRESH_BINARY)[1]) / 255 - 1) * (-1) for k
    #   