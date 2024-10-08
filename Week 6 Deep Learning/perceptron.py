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
with tf.device('/CPU:0'):
  model_0 = models.Sequential([
    layers.Flatten(),
    layers.Dense(10, activation="softmax")
    ])

  model_0.compile(optimizer="adam",
                  loss = "sparse_categorical_crossentropy",
                  metrics=["accuracy"])

  history_0 = model_0.fit(train_images, train_labels, epochs=150,
                        validation_data=(test_images, test_labels))

  model_0_test_loss, model_0_test_acc = model_0.evaluate(test_images, test_labels)

  model_1 = models.Sequential([
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
  ])

  model_1.compile(optimizer = "adam",
                  loss = "sparse_categorical_crossentropy",
                  metrics=["accuracy"])

  history_1 = model_1.fit(train_images, train_labels, epochs=150,
                        validation_data = (test_images, test_labels))

  model_1_test_loss, model_1_test_acc = model_1.evaluate(test_images, test_labels)

  model_2 = models.Sequential([
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
  ])
  model_2.compile(optimizer="adam",
                  loss = "sparse_categorical_crossentropy",
                  metrics=["accuracy"])

  history_2 = model_2.fit(train_images, train_labels, epochs=150,
                        validation_data=(test_images, test_labels))

  model_2_test_loss, model_2_test_acc = model_2.evaluate(test_images, test_labels)

  plt.figure(figsize=(16,10))
  for i, hist in enumerate([history_0, history_1, history_2]):
      plt.plot(hist.history['accuracy'], label=f'accuracy model with {i} hidden layers')
      plt.plot(hist.history['val_accuracy'], label = f'val_accuracy model with {i} hidden layers')
      plt.xlabel('Epoch')
      plt.ylabel('Accuracy')
      plt.legend(loc='lower right')
  plt.show()


  model_20 = models.Sequential([
    layers.Flatten(),
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
    ])

  model_20.compile(optimizer="adam",
                  loss = "sparse_categorical_crossentropy",
                  metrics=["accuracy"])

  history_20 = model_20.fit(train_images, train_labels, epochs=1000,
                        validation_data=(test_images, test_labels))

  model_20_test_loss, model_20_test_acc = model_20.evaluate(test_images, test_labels)

  plt.figure(figsize=(16,10))
  plt.plot(history_20.history['accuracy'], label='accuracy')
  plt.plot(history_20.history['val_accuracy'], label = 'val_accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(loc='lower right')
  plt.show()