#neural network classification

#spam or not spam - binary
#multiclass classification
#multilabel classification

#neural network classification model, harnessing the power of non-linearity and different classification methods

import torch
from torch import nn
import sklearn
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt

def chapter_2_notes():
    n_samples = 1000
    X, y = make_circles(n_samples, noise=0.03, random_state=42)

    # print(f"X: {X[:5]}")

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

    ### turn data into tensors

    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y).type(torch.float)

    #split data into training and test sets

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class CircleModelV0(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Linear(in_features=2, out_features=5)
            self.layer_2 = nn.Linear(in_features=5, out_features=1)
        
        def forward(self, x):
            return self.layer_2(self.layer_1(x))
        
    model_0 = CircleModelV0().to(device)

    # print(model_0)

    # using nn.Sequential to make same model because model is quite simple

    model_0 = nn.Sequential(
        nn.Linear(in_features=2, out_features=5),
        nn.Linear(in_features=5, out_features=1)
    ).to(device)

    #OR

    class CircleModelV01(nn.Module):
        def __init__(self):
            super().__init__()
            self.two_layer = nn.Sequential(
                nn.Linear(in_features=2, out_features=5),
                nn.Linear(in_features=5, out_features=1)
            ) 
        
        def forward(self, x):
            return self.two_layer(x)
        
    model_01 = CircleModelV01().to(device)

    # print(model_01.state_dict())
    # print(model_0.state_dict())

    with torch.inference_mode():
        untrained_preds = model_0(X_test.to(device))

    loss_fn = nn.BCEWithLogitsLoss() # includes sigmoid layer

    optimiser = torch.optim.SGD(params=model_0.parameters(),
                                lr=0.1)

    def accuracy_fn(y_true, y_pred):
        correct  = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100
        return acc

    #training a model, need a loop

    with torch.inference_mode():
        y_logits = model_0(X_test.to(device))[:5]

    # print(untrained_preds[:5])
    # print(y_logits[:5])

    y_pred_probs = torch.sigmoid(y_logits) #output model into probabilities

    # torch.round(y_pred_probs)

    y_preds = torch.round(y_pred_probs)
    y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))
    # print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))
    # print(y_preds.squeeze())
    # print(y_test[:5])

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    epochs = 100

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    for epoch in range(epochs):
        model_0.train()

        y_logits = model_0(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))

        #for BCELoss (not with logits)
        # loss = loss_fn(torch.sigmoid(y_logits),
        #                y_train)

        loss = loss_fn(y_logits,
                    y_train)
        
        acc = accuracy_fn(y_true=y_train,
                        y_pred=y_pred)
        
        optimiser.zero_grad()

        loss.backward()

        optimiser.step()

        model_0.eval()
        with torch.inference_mode():
            test_logits = model_0(X_test).squeeze()

            test_pred = torch.round(torch.sigmoid(test_logits))

            test_loss = loss_fn(test_logits,
                                y_test)
            
            test_acc = accuracy_fn(y_test, test_pred)

        # if epoch % 10 == 0:
        #     print(f"Epoch: {epoch} | Loss: {loss:.05f}, Acc: {acc:0.2f}%  | Test loss: {test_loss:0.5f}, Test Acc: {test_acc:0.2f}%")

    import requests
    from pathlib import Path

    # if Path("helper_functions.py").is_file():
    #     print("already exists")
    # else:
    #     print("downloading helper function")
    #     request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    #     with open("helper_functions.py", "wb") as f:
    #         f.write(request.content)
        
    from helper_functions import plot_predictions, plot_decision_boundary

    def plot_train_test():
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Train")
        plot_decision_boundary(model_0, X_train, y_train)
        plt.subplot(1, 2, 2)
        plt.title("Test")
        plot_decision_boundary(model_0, X_test, y_test)
        plt.show()

        return 0

    #improving a model

    class CircleModelV1(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Linear(in_features=2, out_features=10)
            self.layer_2 = nn.Linear(in_features=10, out_features=10) # extra layer
            self.layer_3 = nn.Linear(in_features=10, out_features=1)
            
        def forward(self, x): # note: always make sure forward is spelt correctly!
            # Creating a model like this is the same as below, though below
            # generally benefits from speedups where possible.
            # z = self.layer_1(x)
            # z = self.layer_2(z)
            # z = self.layer_3(z)
            # return z
            return self.layer_3(self.layer_2(self.layer_1(x)))

    model_1 = CircleModelV1().to(device)

    loss_fn = nn.BCEWithLogitsLoss()

    optimiser = torch.optim.SGD(model_1.parameters(),
                                lr = 0.1)

    torch.manual_seed(42)

    epochs = 1000 # Train for longer

    # Put data to target device
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    for epoch in range(epochs):
        ### Training
        # 1. Forward pass
        y_logits = model_1(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> predicition probabilities -> prediction labels

        # 2. Calculate loss/accuracy
        loss = loss_fn(y_logits, y_train)
        acc = accuracy_fn(y_true=y_train, 
                        y_pred=y_pred)

        # 3. Optimizer zero grad
        optimiser.zero_grad()

        # 4. Loss backwards
        loss.backward()

        # 5. Optimizer step
        optimiser.step()

        ### Testing
        model_1.eval()
        with torch.inference_mode():
            # 1. Forward pass
            test_logits = model_1(X_test).squeeze() 
            test_pred = torch.round(torch.sigmoid(test_logits))
            # 2. Caculate loss/accuracy
            test_loss = loss_fn(test_logits,
                                y_test)
            test_acc = accuracy_fn(y_true=y_test,
                                y_pred=test_pred)

        # Print out what's happening every 10 epochs
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")


    #non-linearity

    class CircleModelV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Linear(in_features=2, out_features=10)
            self.layer_2 = nn.Linear(in_features=10, out_features=10)
            self.layer_3 = nn.Linear(in_features=10, out_features=1)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
        

    model_3 = CircleModelV2()

    loss_fn = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.SGD(model_3.parameters(), lr=0.1)

    torch.manual_seed(42)

    torch.cuda.manual_seed(42)

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    epochs = 1000

    for epoch in range(epochs):
        model_3.train()

        y_logits = model_3(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))

        loss = loss_fn(y_logits, y_train)
        acc = accuracy_fn(y_train, y_pred)

        optimiser.zero_grad()

        loss.backward()

        optimiser.step()

        model_3.eval()

        with torch.inference_mode():
            test_logits = model_3(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))

            test_loss = loss_fn(test_logits, y_test)
            test_accuracy = accuracy_fn(y_test, test_pred)

        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss}, Acc: {acc:.2f}% | Test loss: {test_loss:0.5f}")
    return 0


# def plot():
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.title("Train")
#     plot_decision_boundary(model_1, X_train, y_train) # model_1 = no non-linearity
#     plt.subplot(1, 2, 2)
#     plt.title("Test")
#     plot_decision_boundary(model_3, X_test, y_test)
#     plt.show()
# plot()

A = torch.arange(-10, 10, 1, dtype=torch.float32)

def relu(x): # returns max of 0 and x
    return torch.maximum(torch.tensor(0), x)


## multi-class classification problem

#data

def accuracy_fn(y_true, y_pred):
        correct  = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100
        return acc























