#!/usr/bin/python

from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Activation, Dropout, Reshape, merge
from keras.models import model_from_json
from termcolor import colored
import numpy as np
import os
import cv2
import csv
import h5py
import sys

sys.path.insert(0, 'src/classifiers/')
import Utils
from skimage import feature

WIDTH = 90 # 180 for captcha6_level2/4
HEIGHT = 100

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

input_file = csv.reader(open('src/preprocessing/all_labels_old.txt', 'rt'), delimiter=';')
count = 0.0
lines = 0.0

for line in input_file:
    name = 'datasets/facesInTheWild/%s' % str(line[0]).replace('\n', '').replace('\r', '')

    img = cv2.imread(name, 0)
    img = Utils.cropToFace(img)
    feat = feature.local_binary_pattern(img, 24, 3, method="default")
    img = np.resize(feat.astype('uint8'), (HEIGHT, WIDTH, 3))
    img = np.multiply(img, 1 / 255.0)
    chal_img = np.asarray(img)
    resized_image = np.expand_dims(chal_img, axis=0)
    out = loaded_model.predict(resized_image)
    best_guess = np.argmax(out)


    lines += 1.0
    if a_array[best_guess] == str(line[1]).replace('\n', '').replace('\r', '').replace(' ', ''):
        count += 1.0
    else:
        print(line[0] + ': PRED: ' + a_array[best_guess] + ' REAL: ' + line[1])

print('accuracy : ' + str(((count / lines) * 100.0)))
