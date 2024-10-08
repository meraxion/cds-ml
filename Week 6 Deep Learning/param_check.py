# Adapted from the code on https://www.tensorflow.org/tutorials/images/cnn 
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

model_0 = models.Sequential([
  layers.Flatten(),
  layers.Dense(10, activation="softmax")
  ])

model_0.compile(optimizer="adam",
                loss = "sparse_categorical_crossentropy",
                metrics=["accuracy"])

model_0.build(train_images.shape)
model_0.summary()

model_1 = models.Sequential([
  layers.Flatten(),
  layers.Dense(64, activation="relu"),
  layers.Dense(10, activation="softmax")])

model_1.compile(optimizer="adam",
                loss = "sparse_categorical_crossentropy",
                metrics=["accuracy"])

model_1.build(train_images.shape)
model_1.summary()

model_2 = models.Sequential([
  layers.Flatten(),
  layers.Dense(64, activation="relu"),
  layers.Dense(64, activation="relu"),
  layers.Dense(10, activation="softmax")
])

model_2.compile(optimizer="adam",
                loss = "sparse_categorical_crossentropy",
                metrics=["accuracy"])

model_2.build(train_images.shape)
model_2.summary()

model_20 = models.Sequential([
  layers.Flatten(),
  layers.Dense(512, activation="relu"),
  layers.Dense(10, activation="softmax")
])

model_20.compile(optimizer="adam",
                loss = "sparse_categorical_crossentropy",
                metrics=["accuracy"])

model_20.build(train_images.shape)
model_20.summary()