import numpy as np
import pandas as pd


class summary():
    def __init__(self, model, optimizer, learning_rate, regularization, lbd, iters):
        self.model = model
        self.optimizer = optimizer
        self.specif = ''
        self.costs = []
        self.train_time = 0
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.lbd = lbd
        self.iterations = iters
        self.train_accuracy = 0
        self.test_accuracy = 0
        self.train_size = 0
        self.dataset = None

    def to_dict(self):
        return dict((name, [getattr(self, name)]) for name in vars(self) if not name.startswith('costs'))

    def save_to_csv(self, filename='summary.csv'):
        sum_dict = self.to_dict()

        df = pd.DataFrame.from_dict(sum_dict)

        df.to_csv(filename, mode='a', header=False)

    def set_accuracies(self, pred_train, pred_test, y_train, y_test, print_acc=False):
        self.test_accuracy = 100 - np.mean(np.abs(pred_test - y_test)) * 100
        self.train_accuracy = 100 - np.mean(np.abs(pred_train - y_train)) * 100

        if print_acc:
            print("train accuracy: {} %".format(self.train_accuracy))
            print("train accuracy: {} %".format(self.test_accuracy))
