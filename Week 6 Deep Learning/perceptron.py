# Adapted from the code on https://www.tensorflow.org/tutorials/images/cnn 
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# model = models.Sequential([
#     layers.Flatten(),
#     layers.Dense(10, activation='softmax')])
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# history = model.fit(train_images, train_labels, epochs=50,
#                     validation_data=(test_images, test_labels))

# test_loss, test_acc = model.evaluate(test_images, test_labels)

model_0 = models.Sequentual([
  layers.Flatten(),
  layers.Dense(10, activation="softmax")
  ])

model_0.compile(optimizer="adam",
                loss = "sparse_categorical_crossentropy",
                metrics=["accuracy"])

history_0 = model_0.fit(train_images, train_labels, epochs=50,
                      validation_data=(test_images, test_labels))

model_0_test_loss, model_0_test_acc = model_0.evaluate(test_images, test_labels)

model_1 = models.Sequential([
  layers.Flatten(),
  layers.Dense(64, activation="ReLU"),
  layers.Dense(10, activation="softmax")
])

model_1.compile(optimizer = "adam",
                loss = "sparse_categorical_crossentropy",
                metrics=["accuracy"])

history_1 = model_1.fit(train_images, train_labels, epochs=50,
                      validation_data = (test_images, test_labels))

model_1_test_loss, model_1_test_acc = model_1.evaluate(test_images, test_labels)

model_2 = models.Sequential([
  layers.Flatten(),
  layers.Dense(64, activation="ReLU"),
  layers.Dense(64, activation="ReLU"),
  layers.Dense(10, activation="softmax")
])
model_2.compile(optimizer="adam",
                loss = "sparse_categorical_crossentropy",
                metrics=["accuracy"])

history_2 = model_2.fit(train_images, train_labels, epochs=50,
                      validation_data=(test_images, test_labels))

model_2_test_loss, model_2_test_acc = model_2.evaluate(test_images, test_labels)


for history in [history_0, history_1, history_2]:
    plt.plot(history.history['acc'], label='accuracy')
    plt.plot(history.history['val_acc'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')