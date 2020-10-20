#  -*- coding: utf-8 -*-
#  @Author: Victor Garcia
#  @Date:   2018-10-02 11:16:43
#  @Last Modified by:   Victor Garcia
#  @Last Modified time: 2018-11-08 10:02:28
import pyximport
import sys
import argparse
import numpy as np
import os
import cv2
import itertools
from datetime import datetime
from utils.model_preprocessing import save_model, get_model_callbacks, \
                                     save_train_history, load_model
from aux_scripts.metrics_calculator import compute_metrics
from keras.models import Model
from keras.layers import Activation, Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import keras.backend as K

#  Avoid unnecesary warning during training
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pyximport.install()
"""
DATASET PROCESSING
"""


def load_segmentation_data(dataset_folder, img_width, img_height,
                           batch_size, validation_split=0.1, augmented=False,
                           train=False, test=False, train_test=False):
    """
    Generator for training.
    As we do not need labels, for training all class mode are None
    every folder must have a subfolder (images/) containing the images
    due to Keras API functioning
    3 generators are needed: 2 for input images and one for output.
    We have to repeat these for test and training (6 generators as output).
    IF SEEDS ARE EQUAL RANDOMIZATION ARE THE SAME
    """
    #  Augmented/non-augmented dataset
    if augmented is True:
        datagen = ImageDataGenerator(rescale=1./255, rotation_range=0.2,
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     validation_split=validation_split)
    else:
        datagen = ImageDataGenerator(rescale=1./255,
                                     validation_split=validation_split)
    """
    TRAINING
    """
    if train is True or train_test is True:
        gen_train_in1 = datagen.flow_from_directory(
                dataset_folder+'fore/', class_mode=None,
                batch_size=batch_size, target_size=(img_width, img_height),
                shuffle=True, seed=1, subset='training')
        gen_train_in2 = datagen.flow_from_directory(
                dataset_folder+'back/', class_mode=None,
                batch_size=batch_size, target_size=(img_width, img_height),
                shuffle=True, seed=1, subset='training')
        gen_train_out = datagen.flow_from_directory(
                dataset_folder+'groundtruth/', color_mode='grayscale',
                class_mode=None, batch_size=batch_size,
                target_size=(img_width, img_height), shuffle=True,
                seed=1, subset='training')
    if train is True:
        return gen_train_in1, gen_train_in2, gen_train_out
    """
    TEST
    """
    if test is True or train_test is True:

        gen_test_in1 = datagen.flow_from_directory(
                dataset_folder+'fore/', class_mode=None,
                batch_size=batch_size, target_size=(img_width, img_height),
                shuffle=True, seed=1, subset='validation')

        gen_test_in2 = datagen.flow_from_directory(
                dataset_folder+'back/', class_mode=None, batch_size=batch_size,
                target_size=(img_width, img_height), shuffle=True,
                seed=1, subset='validation')

        gen_test_out = datagen.flow_from_directory(
                dataset_folder+'groundtruth/', color_mode='grayscale',
                class_mode=None, batch_size=batch_size,
                target_size=(img_width, img_height), shuffle=True,
                seed=1, subset='validation')

    if test is True:
        return gen_test_in1, gen_test_in2, gen_test_out
    if train_test is True:
        return gen_train_in1, gen_train_in2, gen_train_out,
    gen_test_in1, gen_test_in2, gen_test_out
    if train is not True and test is not True and train_test is not True:
        print('You must specify the generators mode: \
              train, test, or test_train')
        sys.exit(1)


def create_generator(x1_gen, x2_gen, y_gen=None):
    """
    This script generates the cnn and trains the model or
    segmentation_cd model (mask output)
    Generator creation for model's training.
    Two images as input, one image as output
    """
    x1_batch = np.array([])
    x2_batch = np.array([])
    if y_gen is not None:
        y_batch = np.array([])
        # iterates over the 3 generators
        for x1_batch, x2_batch, y_batch in \
                itertools.zip_longest(x1_gen, x2_gen, y_gen):
            y_batch_final = []
            # concatenates both input in depth axis
            x_batch = np.concatenate((x1_batch, x2_batch), axis=3)
            # append each output (mask) image as an array
            # (64x64 image=> 4096 length vector)
            for index, value in enumerate(y_batch):
                y_batch_final.append(y_batch[index].flatten())
            y_batch_final = np.array(y_batch_final)
            # returns the input, output batch
            yield(x_batch, y_batch_final)
    else:
        for x1_batch, x2_batch in itertools.zip_longest(x1_gen, x2_gen):
            # concatenates both input in depth axis
            x_batch = np.concatenate((x1_batch, x2_batch), axis=3)
            # append each output (mask) image as an array
            # (64x64 image=> 4096 length vector)
            # returns the input, output batch
            yield(x_batch)


def generate_model(img_width, img_height):
    input_shape = (img_width, img_height, 6)
    image_input = Input(shape=input_shape)
    # In functional Keras model, the previous layer must be specified
    # after the layer declarations
    # CNN1
    current_layer = Conv2D(24, (3, 3), padding='same')(image_input)
    current_layer = Activation('relu')(current_layer)
    current_layer = MaxPooling2D(pool_size=(2, 2))(current_layer)
    # CNN2
    current_layer = Conv2D(48, (3, 3), padding='same')(current_layer)
    current_layer = Activation('relu')(current_layer)
    current_layer = MaxPooling2D(pool_size=(2, 2))(current_layer)
    # CNN3
    current_layer = Conv2D(96, (3, 3), padding='same')(current_layer)
    current_layer = Activation('relu')(current_layer)
    current_layer = MaxPooling2D(pool_size=(2, 2))(current_layer)
    # CNN4
    current_layer = Conv2D(96, (3, 3), padding='same')(current_layer)
    current_layer = Activation('relu')(current_layer)
    current_layer = MaxPooling2D(pool_size=(2, 2))(current_layer)
    # Flatten and dense (output vector, with binary values)
    # Sigmoid function as we need values
    # between 0 and 1 in mask (normalized image)
    current_layer = Flatten()(current_layer)
    current_layer = Dropout(0.5)(current_layer)
    current_layer = BatchNormalization()(current_layer)
    final_output = Dense(
            4096, activation='sigmoid', name='final_sigmoid')(current_layer)
    # declare model with input and output layer (Keras Functional format)
    final_model = Model(inputs=image_input, outputs=final_output)
    # prints model's layers as well as its parameters.
    print(final_model.summary())
    return final_model


if __name__ is "__main__":
    K.clear_session()
    parser = argparse.ArgumentParser(description='Process a video.')
    parser.add_argument('dataset_folder', metavar='dataset_folder', type=str,
                        help='path to dataset_folder')
    parser.add_argument('--mode', '-m', type=str, default='train',
                        help='test or train mode specification')
    parser.add_argument('--test_split', '-t', type=float, default=0.1,
                        help='set validation split parameter')
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('--augmented', action='store_true',
                        help='select augmented dataset format or not')
    parser.add_argument('--monitor', type=str, default='acc',
                        help='select the improvement monitor of model:\
                        val_acc, acc, val_loss, loss, \
                        or any other metric specified for the model')
    args = parser.parse_args()
    # loads and image from the dataset
    # in order to obtain the model input dimensions
    subset_folder = os.listdir(
            os.path.abspath(args.dataset_folder+'/back/'))[0]
    test_image_folder = os.path.abspath(
            args.dataset_folder+'/back/'+subset_folder)
    test_image = cv2.imread(
            test_image_folder+'/'+os.listdir(test_image_folder)[1])
    img_width, img_height = test_image.shape[1], test_image.shape[0]
    gen_test_in1, gen_test_in2, gen_test_out = \
        load_segmentation_data(
                    args.dataset_folder, img_width,
                    img_height, args.batch_size,
                    args.test_split, args.augmented, test=True)
    if args.mode is 'train':
        model = generate_model(img_width, img_height)
        gen_train_in1, gen_train_in2, gen_train_out,\
            gen_test_in1, gen_test_in2, gen_test_out = \
            load_segmentation_data(
                    args.dataset_folder, img_width, img_height,
                    args.batch_size, args.test_split,
                    args.augmented, train_test=True)
        model_optimizer = Adam(lr=0.0001)
        model.compile(loss='binary_crossentropy',
                      optimizer=model_optimizer, metrics=['accuracy'])
        # Structure for model creation (and save)
        model_date = "{:%d_%m-%H_%M}".format(datetime.now())
        model_path = 'models_segmentation_cd/'+model_date
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        callbacks_list = get_model_callbacks(model_path, args.monitor)
        save_model(model, model_path)
        train_history = model.fit_generator(
            create_generator(gen_train_in1, gen_train_in2, gen_train_out),
            steps_per_epoch=len(gen_train_in1),
            epochs=args.epochs,
            validation_data=create_generator(
                    gen_test_in1, gen_test_in2, gen_test_out),
            validation_steps=len(gen_test_in1),
            workers=1,
            use_multiprocessing=False,
            verbose=1,
            callbacks=callbacks_list)
        save_train_history(train_history)
    elif args.mode is 'test':
        model = load_model('models_segmentation_cd/25_09-11_22/')
        y_pred = model.predict_generator(
                create_generator(gen_test_in1, gen_test_in2),
                steps=len(gen_test_in1), verbose=1)
        # print(y_pred.shape) # 5375, 4096
        compute_metrics(y_pred, gen_test_out, img_width, img_height)
    # Clears TensorFlow session to avoid conflicts in future trainings
    K.clear_session()
