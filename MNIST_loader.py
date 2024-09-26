#
# This is a sample Notebook to demonstrate how to read "MNIST Dataset"
#
import numpy as np  # linear algebra
import struct
from array import array
from os.path import join


#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)
    #


# Verify Reading Dataset via MnistDataloader class
#
import random
import matplotlib.pyplot as plt

#
# Set file paths based on added MNIST Datasets
#
input_path = '../MNIST'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')


#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize=15);
        index += 1

    plt.show()


def normalize(X):
    return X / 255


def denormalize(X):
    return X * 255


def relabel(Y, mapping):
    return np.fromiter((mapping.get(x, x) for x in Y), dtype=int)


def preprocess(X, Y):
    # process Y
    Y = np.array(Y)
    relevant_numbers_idx = np.where((Y == 3) | (Y == 7))
    Y_filter = Y[relevant_numbers_idx[0]]
    mapping = {3: 0, 7: 1}
    Y_normalized = relabel(Y_filter, mapping)

    # process X
    X = np.array(X)
    X_filter = X[relevant_numbers_idx[0], :, :]
    X_normalized = normalize(X_filter)
    X_normalized = X_normalized.reshape(len(Y_normalized), X_normalized.shape[1] * X_normalized.shape[2])
    return X_normalized, Y_normalized, X_filter, Y_filter


#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                   test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

X_train_norm, Y_train_norm, X_train_raw, Y_train_raw = preprocess(x_train, y_train)
np.savez('train_data.npz',
         X_train_norm=X_train_norm,
         Y_train_norm=Y_train_norm,
         X_train_raw=X_train_raw,
         Y_train_raw=Y_train_raw)
# np.savez('train_data.npz', train_data=train_data)
# X_train_norm, Y_train_norm, X_train_raw, Y_train_raw = train_data

X_test_norm, Y_test_norm, X_test_raw, Y_test_raw = preprocess(x_test, y_test)
np.savez('test_data.npz',
         X_test_norm=X_test_norm,
         Y_test_norm=Y_test_norm,
         X_test_raw=X_test_raw,
         Y_test_raw=Y_test_raw)
# np.savez('test_data.npz', train_data=test_data)
# X_test_norm, Y_test_norm, X_test_raw, Y_test_raw = test_data

#
# Show some random training and test images
#
images_2_show = []
titles_2_show = []
for i in range(0, 10):
    r = random.randint(1, len(X_train_raw))
    images_2_show.append(X_train_raw[r])
    titles_2_show.append('training image [' + str(r) + '] = ' + str(Y_train_norm[r]))

for i in range(0, 5):
    r = random.randint(1, len(X_test_raw))
    images_2_show.append(X_test_raw[r])
    titles_2_show.append('test image [' + str(r) + '] = ' + str(Y_test_norm[r]))

show_images(images_2_show, titles_2_show)
