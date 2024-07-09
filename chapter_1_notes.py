import torch
from torch import nn
import matplotlib.pyplot as plt

#linear regression to make a straight line with known parameters 

def linearRegressions():

    weights = 0.7
    bias = 0.3

    start = 0
    end = 1
    step = 0.02
    X = torch.arange(start, end, step).unsqueeze(dim=1)

    y = weights * X + bias

    # print(X[:10], y[:10])


    train_split = int(0.8 * len(X))
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    # print(train_split)
    # print(X_train)
    # print(X_test)

    
    return 0

# linearRegressions()

#training and test sets






