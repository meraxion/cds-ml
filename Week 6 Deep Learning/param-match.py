# Adapted from the code on https://www.tensorflow.org/tutorials/images/cnn 
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers, optimizers
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# num params in our cnn model: 122_570

model_match = models.Sequential([
  layers.Flatten(),
  layers.Dense(40, activation = "relu"),
  layers.Dense(10, activation = "softmax")
])

model_match.compile(optimizer = optimizers.Adam(learning_rate=1e-3, weight_decay=3.0), loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

model_match.build(train_images.shape)
model_match.summary()

history = model_match.fit(train_images, train_labels, epochs=25,
                        validation_data = (test_images, test_labels))