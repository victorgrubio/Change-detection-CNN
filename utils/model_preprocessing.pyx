# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-09-25 09:35:30
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-09-25 09:35:30
import json
import cv2
import numpy as np
cimport numpy as np
import json
from datetime import datetime
from sys import exit
from keras.models import load_model, model_from_json
from keras.utils import plot_model
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
import keras.backend as K


"""
This class contains methods used in multiple scripts in order to compact
the code and avoid repetition related to predicts and model processing
"""


def get_predict_img(img_shape, predicts_array, window_size, step):
    """
    Returns the output img as a union of predicted patches stored
    in predicts_array. Useful for SEGMENTATION and reconstruction
    """
    cdef np.ndarray predict_img = np.zeros((img_shape), dtype=np.int)
    # length of step in px
    cdef int step_size = window_size // step
    cdef int y_limit = img_shape[0]
    cdef int x_limit = img_shape[1]
    # number of windows for each img dimension
    cdef int y_windows = img_shape[0] // step_size - step
    cdef int x_windows = img_shape[1] // step_size - step
    # variable declaration for cython
    cdef int x, y, index
    cdef int counter_y_windows = 0
    cdef int counter_x_windows = 0
    # construct predict img from predict array structure
    for index in range(len(predicts_array)):
        y = counter_y_windows * step_size
        x = counter_x_windows * step_size
        # do not extract img if we are out of img dimensions
        if y + window_size <= y_limit and\
                x + window_size <= x_limit:
            predict_img[
                y:y + window_size, x:x + window_size] += \
                predicts_array[index]
        if counter_x_windows < x_windows:
            counter_x_windows += 1
        # if we are in last window, we shift to the following row
        elif counter_y_windows < y_windows:
            counter_x_windows = 0
            counter_y_windows += 1
    # divide values with the number of overlapping regions (step²)
    predict_img //= (step**2)
    # change to img format in order to use imshow
    predict_img = predict_img.astype(np.uint8)
    return predict_img


def predict_batches(input_array, model, int batch_size):
    """
    Predict using batches, iterating through array
    If used for an img  with sliding window,
    batch size should be equal to the number of windows for
    each row (number of divisions in one dimesion)
    """
    cdef int steps = input_array.shape[0] // batch_size + 1
    cdef int step
    batch_predict_array = []
    for step in range(steps):
        # if we are not in last batch
        if step != steps - 1:
            input_batch = input_array[
                step * batch_size:(step + 1) * batch_size]
        # if we are, just go until end of array, do not care about batch
        # as the division probably do not produce and integer value
        else:
            input_batch = input_array[step * batch_size:]
        # predict batch, append to array an show its shape
        batch_predict = model.predict(input_batch, batch_size)
        # With the first batch we obtain the windows array shape
        if step == 0:
            batch_predict_array = list(batch_predict)
        else:
            # Concatenate predict batches to the array
            batch_predict_array.extend(batch_predict)
    return batch_predict_array


def patch_predicts(fore_img, back_img, model, step, debug=False):
    """
    slides and img and applies a predict for each path
    get the sliding window's size from the model's input layer
    specifies an step to overlap windows
    """
    cdef int model_input_size = model.layers[0].input_shape[1]
    cdef int step_size = model_input_size // step
    predicts_array = []
    concatenated_input_array = []
    cdef np.ndarray fore_patch
    cdef np.ndarray back_patch
    cdef np.ndarray fore_input
    cdef np.ndarray back_input
    cdef int y, x
    flag_resize = False
    new_height, new_width = fore_img.shape[0], fore_img.shape[1]
    # Resize imgs if they don't fit the step size before the sliding window
    if fore_img.shape[0] % model_input_size != 0:
        new_height = (fore_img.shape[0] // model_input_size) * model_input_size
        flag_resize = True
    if fore_img.shape[1] % model_input_size != 0:
        new_width = (fore_img.shape[1] // model_input_size) * model_input_size
        flag_resize = True
    if flag_resize is True:
        fore_img = cv2.resize(fore_img, (new_width, new_height),
                              interpolation=cv2.INTER_CUBIC)
        back_img = cv2.resize(back_img, (new_width, new_height),
                              interpolation=cv2.INTER_CUBIC)
    # Set limits for the sliding window algorithm after the resize process
    cdef int width_limit = fore_img.shape[0] - (model_input_size - step_size)
    cdef int height_limit = fore_img.shape[1] - (model_input_size - step_size)
    # slide the img
    for y in range(0, width_limit, step_size):
        for x in range(0, height_limit, step_size):
            # get the patches from foreground and background img
            # applies necessary preprocessing for model predict
            fore_patch = fore_img[y:y + model_input_size,
                                  x:x + model_input_size]
            back_patch = back_img[y:y + model_input_size,
                                  x:x + model_input_size]
            fore_input = prepare_img_predict(fore_patch)
            back_input = prepare_img_predict(back_patch)
            concatenated_input = np.concatenate(
                (fore_input, back_input), axis=2)
            concatenated_input_array.append(concatenated_input)
    """
    Batch predict:
    As we have lots of windows to predict, we have to stablish a batch size
    to predict them using batches of data. This method returns the
    concatenated array of predicts resulting from each batch.
    """
    concatenated_input_nparray = np.array(concatenated_input_array)
    predicts_array = predict_batches(
        concatenated_input_nparray, model, batch_size=32)
    if debug:
        cv2.destroyAllWindows()
    return predicts_array

def array_to_patch(predicts_array, patch_shape, debug=False):
    """
    Method that modifies the obtain predict to have an img
    of 64x64 from a 4096 vector with values [0, 1]
    """
    cdef np.ndarray predict
    cdef int index
    # Update predict with array of 4096 element structure
    for index in range(len(predicts_array)):
        predict = predicts_array[index] * 255
        predict = predict.astype("uint8")
        predict = np.reshape(predict, patch_shape)
        predicts_array[index] = predict
    return predicts_array


def prepare_img_predict(img):
    """
    Expand dims and normalizes img for model predict
    """
    array = np.asarray(img, dtype='int32')
    array = array / 255
    return array


def load_model(model_path):
    """
    Loads model from json and h5py files
    """
    with open(model_path + 'model_architecture.json') as json_data:
        model_name_json = json.load(json_data)
        json_data.close()
    model = model_from_json(model_name_json)
    model.load_weights(model_path + 'model_weights.h5py')
    return model


def save_model(model, model_path):
    """
    Saves model with its layers in a png file
    Need function before fitting models.
    """
    model_json = model.to_json()
    with open(model_path + '/model_architecture.json', 'w') as outfile:
        json.dump(model_json, outfile)
    plot_model(model, to_file=model_path + '/model_layers.png',
               show_shapes=True,
               show_layer_names=True)


def get_model_callbacks(model_path, monitor):
    """
    Generate model callback for training
    Select the mode of monitor: maximize accuracy or minimize loss/error
    """
    if 'acc' in monitor:
        checkpoint_mode = 'max'
    elif 'loss' or 'mean' in monitor:
        checkpoint_mode = 'min'
    checkpoint_callback = ModelCheckpoint(model_path + '/model_weights.h5py',
                                          monitor=monitor, verbose=1,
                                          save_best_only=True,
                                          mode=checkpoint_mode)
    tboard_callback = TensorBoard(log_dir='./tb_graphs',
                                  histogram_freq=0, write_graph=True,
                                  write_imgs=True, update_freq='epoch')
    csv_callback = CSVLogger('{}/training_results.csv'.format(
        model_path), separator=',', append=True)
    return [checkpoint_callback, tboard_callback, csv_callback]


def save_train_history(train_history, model_path):
    with open(model_path + '/train_history.json', 'w') as f:
        json.dump(train_history.history, f)
