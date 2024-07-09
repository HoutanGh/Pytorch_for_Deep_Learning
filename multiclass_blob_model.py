from torch import nn
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from chapter_2_notes import accuracy_fn



NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES,
                            centers=NUM_CLASSES,
                            cluster_std=1.5,
                            random_state=RANDOM_SEED)

X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

# print(y_blob.shape)

X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob, 
                                                                        y_blob,
                                                                        test_size=0.2,
                                                                        random_state=RANDOM_SEED)

# plt.figure(figsize=(10, 7))
# plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
# plt.show()

device = "cuda" if torch.cuda.is_available() else "cpu"

class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()

        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)            
        )

    def forward(self, x):
        return self.linear_layer_stack(x)

#to know the number of output features, print out the len(torch.unique(y_blob_train)) to see the number of classes

model = BlobModel(2, 4, 8).to(device)

# print(model)

loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(),
                            lr=0.1)

#raw outputs of model are logits
#need to convert our model's outputs (logits) to prediction probalities and then to prediction labels using an activation function

# model.eval()
# with torch.inference_mode():
#     y_logits = model(X_blob_test.to(device))
#     # print(y_preds)

# y_pred_probs = torch.softmax(y_logits, dim=1) #dim means the dimension on which the softmax will be applied
# # print(y_logits.shape)
# # print(y_pred_probs[:5])

# y_preds = torch.argmax()

#training loop

torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 100

X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

for epoch in range(epochs):
    model.train()

    y_logits = model(X_blob_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_true=y_blob_train,
                      y_pred=y_pred)
    
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
    model.eval()
    with torch.inference_mode():
        test_logits = model(X_blob_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)

        test_loss = loss_fn(test_logits, test_pred)
        test_acc = accuracy_fn(y_true=y_blob_test,
                               y_pred = test_pred)
        
    
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%") 

from helper_functions import plot_decision_boundary

# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title("Train")
# plot_decision_boundary(model, X_blob_train, y_blob_train)
# plt.subplot(1, 2, 2)
# plt.title("Test")
# plot_decision_boundary(model, X_blob_test, y_blob_test)

# plt.show()

