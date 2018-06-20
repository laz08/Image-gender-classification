#!/usr/bin/python

from keras import backend as K
from keras import optimizers
from keras.layers import Input, Dense, Flatten, Activation, Dropout, Reshape
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, merge
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
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
import sys
from skimage import feature
import matplotlib.pyplot as plt


sys.path.insert(0, 'src/classifiers/')
import Utils

WIDTH = 90
HEIGHT = 100
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


def load_images(path, data, ind_gender, train_ratio):
    datas = []
    labels = []
    val_data = []
    val_label = []
    images_path = path + '/facesInTheWild/'
    MAX_COUNT = 3000
    count = 0
    previous_name = ''
    same_instance = 0

    # IMAGES PER INDIVIDUAL
    for i, name in enumerate(data):

        img = cv2.imread(images_path + str(name), 0)
        img = Utils.cropToFace(img)
        # img = cv2.resize(img, (WIDTH, HEIGHT))

        if name == previous_name:
            same_instance += 1

            if same_instance == 1:
                continue
        else:
            count += 1
            same_instance = 0

        feat = feature.local_binary_pattern(img, 24, 3  , method="default")
        img = np.resize(feat.astype('uint8'), (HEIGHT, WIDTH, CHANNEL))
        img = np.multiply(img, 1 / 255.0)
        img = np.asarray(img)
        datas.append(img)
        labels.append(one_hot_encode(ind_gender[i]))
        previous_name = name
    count = 0
    previous_name = ''

    datas_labels = list(zip(datas, labels))
    random.shuffle(datas_labels)
    (datas, labels) = list(zip(*datas_labels))
    size = len(labels)
    train_size = int(size * train_ratio)
    train_datas = np.stack(datas[0: train_size])
    train_labels = np.stack(labels[0: train_size])
    test_datas = np.stack(datas[train_size: size])
    test_labels = np.stack(labels[train_size: size])

    return train_datas, train_labels, test_datas, test_labels, val_data, val_label


def get_cnn_net():
    inputs = Input(shape=(HEIGHT, WIDTH, CHANNEL))

    model = Sequential()
    model.add(Conv2D(32, (5, 5), border_mode='valid', kernel_initializer='glorot_uniform', bias_initializer='zeros', input_shape=(HEIGHT, WIDTH, CHANNEL)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.6))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())

    x = model(inputs)
    print(inputs)
    x1 = Dense(2, activation='softmax')(x)
    x = Reshape((1, 2))(x1)
    model = Model(input=inputs, output=x)

    sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.01, nesterov=True)

    model.compile(loss='binary_crossentropy', loss_weights=[1.], optimizer='adam', metrics=['accuracy'])

    return model


def load_csv(filename):
    lines = csv.reader(open(filename, 'rt'), delimiter=';')
    dataset = []
    labels = []
    for i, row in enumerate(lines):
        dataset.append(str(row[0]))
        labels.append(str(row[1]).replace(' ', ''))
    return dataset, labels


dataset, labels = load_csv('src/preprocessing/merged_labels.txt')
(train_datas, train_labels, test_datas, test_labels, val_datas, val_labels) = load_images('datasets', dataset, labels, 0.8)
model = get_cnn_net()

checkpoint = ModelCheckpoint('captcha6_20k_orig_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=2, verbose=1, mode='auto')
hist = model.fit(train_datas, train_labels, epochs=100, batch_size=32, verbose=1,
          callbacks=[tensorboard, es, checkpoint], validation_split=0.1, shuffle=True)
predict_labels = model.predict(test_datas, batch_size=32)
test_size = len(test_labels)
y1 = test_labels[:, 0, :].argmax(1) == predict_labels[:, 0, :].argmax(1)
acc = (y1).sum() * 1.0

print('\nmodel evaluate:\nacc: ', acc / test_size)
print('y1 ', (y1.sum()) * 1.0 / test_size)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Save model and weights for trained model
# model.save_weights('captcha6_20k_orig_model.h5')
with open('captcha6_20k_orig_model.json', 'w') as f:
    f.write(model.to_json())

K.clear_session()
del sess
