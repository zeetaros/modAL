import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import MNIST

from skorch import NeuralNetClassifier

from modAL import ActiveLearner
from modAL.bayesianDL import max_entropy


class SimpleClassifier(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(SimpleClassifier, self).__init__()

        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Dropout(0.25),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(14*14*32, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.convblock(x)
        x = self.fc(x.view(x.shape[0], -1))
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    # loading the data
    mnist = MNIST('.', download=True, transform=T.ToTensor())
    # defining training data
    labels = np.array([data[1] for data in mnist])
    train_idx = []
    for digit in range(10):
        digit_sampled = np.random.choice(np.where(labels == digit)[0], 10, replace=False)
        train_idx = np.concatenate((train_idx, digit_sampled)).astype(int)

    X_train = np.vstack((mnist[0][0][None, :] for i in train_idx))
    y_train = np.vstack((mnist[i][1] for i in train_idx)).reshape(-1)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = NeuralNetClassifier(SimpleClassifier,
                                module__in_channels=1, module__n_classes=10,
                                criterion=nn.CrossEntropyLoss,
                                device=device)

    learner = ActiveLearner(estimator=model, query_strategy=max_entropy,
                            X_training=X_train, y_training=y_train)

    X_pool = np.vstack((mnist[0][0][None, :] for i in range(100)))
    y_pool = np.vstack((mnist[i][1] for i in range(100))).reshape(-1)

    learner.query(X_pool, n_instances=10)