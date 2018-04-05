#%matplotlib inline
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras import regularizers

from keras.datasets import cifar10

from kerasutils import hits_and_misses, describe_model
from imageutils import block_dct
from imageutils import dataset_transform_block_dct

#import matplotlib.pyplot as plt
import numpy as np

import random

print('Loading CIFAR-10 dataset')
(trainX_orig, trainY_orig), (testX, testY) = cifar10.load_data()

# Have to decimate training set size due to memory constraints?
(trainX, testX_dummy, trainY, testY_dummy) = train_test_split(trainX_orig, trainY_orig, test_size=0.25)

trainX = trainX / 255.0
testX = testX / 255.0
print(trainX.shape)
print(testX.shape)
print(trainX.dtype)

# Preprocess input data
print('DCT Block Transform Input Data')
trainX = dataset_transform_block_dct(trainX)
testX = dataset_transform_block_dct(testX)
print(trainX.shape)
print(testX.shape)

# Transform labels from int to one-hot vectors
print('Transform output to one-hot vectors')
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

print(trainY.shape)
print(testY.shape)
n_classes = trainY.shape[1]
height = trainX.shape[1]
width = trainX.shape[2]
channels = trainX.shape[3]

# "Transform" CNN architecture with Keras
model = Sequential()
model.add(Conv2D(input_shape=(height,width,channels), filters=64,
                 use_bias=True, kernel_size=(8,8), strides=8))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Conv2D(filters=128,use_bias=True, kernel_size=(3,3), strides=2))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(lb.classes_.shape[0], activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

n_epochs = 30
batch_size = 256
print('Training model, epochs={0}, batch_size={1}'.format(n_epochs, batch_size))
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              epochs=n_epochs, batch_size=batch_size)
print('Done!!!')

# Evaluate TEST model class prediction accuracy
print("[INFO] Evaluating TEST performance...")
predictions = model.predict(testX, batch_size=batch_size)
target_names = [str(x) for x in lb.classes_]
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=target_names))

# Evaluate TRAIN model class prediction accuracy
print("[INFO] Evaluating TRAIN performance...")
trainPreds = model.predict(trainX, batch_size=batch_size)
target_names = [str(x) for x in lb.classes_]
print(classification_report(trainY.argmax(axis=1),
                            trainPreds.argmax(axis=1),
                            target_names=target_names))

preds = predictions.argmax(axis=1)
targets = testY.argmax(axis=1)
hits, misses = hits_and_misses(preds, targets)
print('{0} hits, {1} misses ({2}%)'.format(len(hits), len(misses), len(hits)/float(len(predictions))))


