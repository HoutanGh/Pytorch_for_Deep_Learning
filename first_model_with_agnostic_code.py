import torch
from torch import nn
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

#data using linear regression formula

weights = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weights * X + bias

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

len(X_train), len(y_train), len(X_test), len(y_test)

def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    
    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})

    plt.show()


# plot_predictions()

class LinearRegressionModelV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # instead of defining specific parameters (weights and biases) just define a linear layer
        self.linear_layer = nn.Linear(in_features=1, # what data goes in
                                  out_features=1) # what data goes out
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
    
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
# print(model_1, model_1.state_dict())

# print(next(model_1.parameters()).device)

model_1.to(device)

# print(next(model_1.parameters()).device)

#training

loss_fn = nn.L1Loss()

optimiser = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.01)

torch.manual_seed(42)

epochs = 200

#need to put data on the target device 
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
    model_1.train() # training mode, all gradients that are needed are readied to be calculated

    y_pred = model_1(X_train) 

    loss = loss_fn(y_pred, y_train)

    optimiser.zero_grad()

    loss.backward()

    optimiser.step()

    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)
        test_loss = loss_fn(test_pred, y_test)

        # if epoch % 10 == 0:
        #     print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")


#turn into evaluation mode
model_1.eval()

with torch.inference_mode():
    y_preds = model_1(X_test)

# plot_predictions(predictions=y_preds.cpu())

from pathlib import Path
MODEL_NAME = "agnostic model.pth"
MODEL_SAVE_PATH = Path("models") / MODEL_NAME

#dont need to create directory because already have one
def save():
    

    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model_1.state_dict(),
            f=MODEL_SAVE_PATH)
    
    return 0

# save()
loaded_model_1 = LinearRegressionModelV2()

def load():

    loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))

    loaded_model_1.to(device)

    print(loaded_model_1.state_dict())

load()

def check_the_same():
    loaded_model_1.eval()
    with torch.inference_mode():
        loaded_model_1_preds = loaded_model_1(X_test)
    
    print(y_preds == loaded_model_1_preds)

check_the_same()








    
    