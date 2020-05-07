import numpy as np
import h5py


def load_dataset(train_file, test_file):
    train_dataset = h5py.File(train_file, "r")
    # your train set features
    x_train = np.array(train_dataset["train_set_x"][:])
    # your train set labels
    y_train = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File(test_file, "r")
    # your test set features
    x_test = np.array(test_dataset["test_set_x"][:])
    y_test = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    y_train = y_train.reshape((1, y_train.shape[0]))
    y_test = y_test.reshape((1, y_test.shape[0]))

    return x_train, y_train, x_test, y_test, classes
