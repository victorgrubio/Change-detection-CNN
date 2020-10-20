import cv2
import numpy as np
import time
from utils.helpers import prepareImage4Model

image_path = 'files/10_10_fore.jpg'
image = cv2.imread(image_path)
window_size = 64
step_size = window_size // 2
print('image shape:', image.shape)

"""
sliding loop (common method)
"""
start_time = time.time()
counter_window = 0
patches_window = []
# print('rows_slide:', (image.shape[0]-(window_size-step_size)) // step_size)
for y in range(0, image.shape[0]-(window_size-step_size), step_size):
    for x in range(0, image.shape[1]-(window_size-step_size), step_size):
        patch = image[y:y + window_size, x:x + window_size]
        input = prepareImage4Model(patch)
        counter_window += 1
        patches_window.append(patch)
patches1_prep = map(prepareImage4Model, patches_window)
print('Time taken by for loop sliding window: ', time.time()-start_time)
print('Total patches on sliding windows:', counter_window)
"""
numpy split method
"""
# THIS IS FAST AS F. ONLY WORKS IF WINDOW_SIZE = STEP SIZE
# Test step size
start_time_split = time.time()
image_rows_1 = np.split(image, image.shape[0] // window_size, axis=0)
image_rows_2 = np.split(
        image[step_size:image.shape[0]-step_size],
        (image.shape[0]-2*step_size) // window_size, axis=0
)
# print('row shape:', image_rows_1[0].shape)
# print('rows_1', len(image_rows_1))
# print('rows_2', len(image_rows_2))
# print('11+12 = 23')
counter_y = 0
counter_x = 0
counter_patches = 0
for image_rows in [image_rows_1, image_rows_2]:
    for row in image_rows:
        y = counter_y*window_size
        # print((row.shape[1]-2*step_size))
        # print(row[:, step_size:row.shape[1]-step_size, :].shape)
        image_patches1 = np.split(row, row.shape[1] // window_size, axis=1)
        image_patches2 = np.split(
                row[:, step_size:row.shape[1]-step_size, :],
                (row.shape[1]-2*step_size) // window_size, axis=1
        )
        patches1_prep = list(map(prepareImage4Model, image_patches1))
        patches2_prep = list(map(prepareImage4Model, image_patches2))
        # print(len(patches2_prep))
        # for patch in image_patches1+image_patches2:
        #     counter_patches += 1
        #     fore_input = prepareImage4Model(patch)

        # for image_splitted in [
        # image_splitted_1, image_splitted_2,
        # image_splitted_3, image_splitted_4]:
        #     for current_patch in image_splitted:
        x = counter_x*window_size
        #         fore_patch = image[y:y + window_size, x:x + window_size]
        #         back_patch = image[y:y + window_size, x:x + window_size]
        #         fore_input = prepareImage4Model(fore_patch)
        #         back_input = prepareImage4Model(back_patch)
        #         counter_x += 1
        counter_y += 1
print('total_patches:', counter_patches)
print('TIME ON NP SPLIT:', time.time()-start_time_split)
