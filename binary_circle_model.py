import torch
from torch import nn
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

n_samples = 1000

# Circular dataset with specified noise level
X, y = make_circles(n_samples, noise=0.03, random_state=42)

# 
circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y})

# print(circles.head(10))

def plot():
    plt.scatter(x=circles["X1"],
                y=circles["X2"],
                c=y,
                cmap=plt.cm.RdYlBu)
    plt.show()

    return 0

# plot()

X_sample = X[0]
y_sample = y[0]

# print(f"X_sample: {X_sample}, shape of X_sample: {X_sample.shape}")
# print(f"y_sample: {y_sample}, shape of y_sample: {y_sample.shape}")

# Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

device = "cuda" if torch.cuda.is_available() else "cpu"

def accuracy_fn(y_true, y_pred):
    # torch.eq compares two tensors element-wise and returns tensor of the same shape containing boolean values
    correct  = torch.eq(y_true, y_pred).sum().item() # .sum() counts number of True values
                                                     # .item() converts the resulting single-element tensor into scalar 
    acc = (correct / len(y_pred)) * 100
    return acc

import requests
from pathlib import Path

if not Path("helper_functions.py").is_file():
    print("downloading helper function")
    # HTTP Get request to url
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    # open new file named "help_functions.py" in write-binary mode
    with open("helper_functions.py", "wb") as f:
        # write the content of the HTTP reponse to the file
        f.write(request.content)

from helper_functions import plot_decision_boundary, plot_predictions

class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU() # adding non-linearity 
    
    def forward(self, x):
        # data flows through the layers with relu activation applied after first and second layers
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x))))) 
    
model = CircleModel().to(device)

# loss function that combines a sigmoid layer and the binary cross-entropy loss in one single class
# derivative of the sigmoid function and the BCE are closely related so more efficient backpropagation
loss_fn = nn.BCEWithLogitsLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.1)
    
torch.manual_seed(42)
torch.cuda.manual_seed(42)

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

epochs = 1000

for epoch in range(epochs):

    # perform forward pass to get raw logits (pre-sigmoid outputs) and squeeze removes dimensions of size 1
    y_logits = model(X_train).squeeze() 

    # torch.sigmoid converts logits to probabilities
    # torch.round rounds probabilities to get binary predictions
    y_pred = torch.round(torch.sigmoid(y_logits))

    # the logits are random float values and are compared to see how close they are to 0 or 1 
    loss = loss_fn(y_logits, y_train)

    # when the logits are rounded, checks if they match
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    optimiser.zero_grad()

    loss.backward()

    optimiser.step()

    model.eval()

    with torch.inference_mode():
        test_logits = model(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_accuracy = accuracy_fn(y_true=y_test, y_pred=test_pred)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:0.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:0.5f}, Test acc: {test_accuracy:0.2f}")

# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title("Train")
# plot_decision_boundary(model, X_test, y_test) # model_1 = no non-linearity
# plt.show()



