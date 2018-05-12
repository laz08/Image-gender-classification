#!/usr/bin/python

from keras import backend as K
from keras.layers import Input, Dense, Flatten, Activation, Dropout, Reshape
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, merge
from keras.callbacks import TensorBoard, EarlyStopping
import keras.metrics
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
import random
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import csv

WIDTH = 250
HEIGHT = 250
CHANNEL = 3

# Optional - this will set so we can release GPU memory after computation
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

seed = 7
np.random.seed(seed)

tensorboard = TensorBoard(log_dir='./Model_graph', histogram_freq=0, write_graph=True, write_images=True)


def categories(label):
    gender_labels = ['MALE', 'FEMALE']

    return gender_labels.index(label)


def one_hot_encode(label):
    return np_utils.to_categorical(np.int32(categories(label)), 2)


def load_images(path, lfw_attributes, train_ratio):
    datas = []
    labels = []
    images_path = path + '/facesInTheWild/'

    # IMAGES PER INDIVIDUAL
    for i, line in enumerate(lfw_attributes):
        name = line[0]
        gender = 'MALE' if float(line[2]) > 0 else 'FEMALE'
        for image_number in range(1, line[1]):
            img = Image.open(images_path + str(name).replace(' ', '_') + '_' + str(image_number).zfill(4) + '.jpg')
            data = img.resize([WIDTH, HEIGHT])
            data = np.multiply(data, 1 / 255.0)
            data = np.asarray(data)
            datas.append(data)
            labels.append(one_hot_encode(gender))
    datas_labels = list(zip(datas, labels))
    random.shuffle(datas_labels)
    (datas, labels) = list(zip(*datas_labels))
    size = len(labels)
    train_size = int(size * train_ratio)
    train_datas = np.stack(datas[0: train_size])
    train_labels = np.stack(labels[0: train_size])
    test_datas = np.stack(datas[train_size: size])
    test_labels = np.stack(labels[train_size: size])

    return (train_datas, train_labels, test_datas, test_labels)


def get_cnn_net():
    inputs = Input(shape=(HEIGHT, WIDTH, CHANNEL))

    model = Sequential()
    model.add(Conv2D(32, (5, 5), border_mode='valid', input_shape=(HEIGHT, WIDTH, CHANNEL)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))
    model.add(Flatten())

    x = model(inputs)
    print(inputs)
    x1 = Dense(2, activation='softmax')(x)
    x = Reshape((1, 2))(x1)
    model = Model(input=inputs, output=x)

    # Visualize model
    plot_model(model, show_shapes=True, to_file='captcha6_20k_orig_graph.png')
    model.compile(loss='categorical_crossentropy', loss_weights=[1.], optimizer='Adam', metrics=['accuracy'])

    return model


def load_csv(filename):
    lines = csv.reader(open(filename, 'rt'), delimiter='\t')
    dataset = []
    for i, row in enumerate(lines):
        if i != 0:
            line = []
            for j, value in enumerate(row):
                column_value = int(value) if value.isdigit() else value
                line.append(column_value)

            if len(line) > 0:
                dataset.append(line)
    return dataset


def main():

    lfw_attributes = load_csv('datasets/lfw_attributes.txt')
    (train_datas, train_labels, test_datas, test_labels) = load_images('datasets', lfw_attributes, 0.8)
    model = get_cnn_net()
    print(model)
    model.fit(train_datas, train_labels, epochs=32, batch_size=32, verbose=1, callbacks=[tensorboard])
    predict_labels = model.predict(test_datas, batch_size=32)
    test_size = len(test_labels)
    y1 = test_labels[:, 0, :].argmax(1) == predict_labels[:, 0, :].argmax(1)
    acc = (y1).sum() * 1.0

    print('\nmodel evaluate:\nacc:', acc / test_size)
    print('y1', (y1.sum()) * 1.0 / test_size)

    # Save model and weights for trained model
    model.save_weights('captcha6_20k_orig_model.h5')
    with open('captcha6_20k_orig_model.json', 'w') as f:
        f.write(model.to_json())

    K.clear_session()
    del sess


if __name__ == "__main__":
   main()
