#!/usr/bin/python

from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Activation, Dropout, Reshape, merge
from keras.models import model_from_json
from termcolor import colored
import numpy as np
import os
import cv2
import h5py

WIDTH = 250 # 180 for captcha6_level2/4
HEIGHT = 250

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load json and create model
json_file = open('captcha6_20k_orig_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)


# load weights into new model
loaded_model.load_weights("captcha6_20k_orig_model.h5")
print("Loaded model from disk")


# print("Treating Zeros and O's fairly - Incorrect predictions are colored white.")

# we have x-number images to predict values for
a_array = ["MALE", "FEMALE"]

input_file = open('datasets/predict_labels.txt')

for i, line in enumerate(input_file):
        chal_img = cv2.imread('datasets/' + str(i))
        resized_image = cv2.resize(chal_img, (WIDTH, HEIGHT)).astype(np.float32)
        resized_image = np.expand_dims(resized_image, axis=0)
        out = loaded_model.predict(resized_image)
        best_guess = np.argmax(out)
        print(best_guess, i)
input_file.close()
