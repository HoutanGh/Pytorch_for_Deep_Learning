import torch
from torch import nn, detach
import matplotlib.pyplot as plt
from pathlib import Path

# Create *known* parameters
weight = 0.7
bias = 0.3

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

X[:10], y[:10]

# Create train/test split
train_split = int(0.8 * len(X)) # 80% of data used for training set, 20% for testing 
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




class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                 requires_grad=True,
                                 dtype=torch.float))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: #overrides nn.Module's own forward function
        return self.weights * x + self.bias

def parametering():
    torch.manual_seed(42)
    model_0 = LinearRegressionModel()
    print(list(model_0.parameters()))

    return 0

torch.manual_seed(42)
model_0 = LinearRegressionModel()

with torch.inference_mode(): # turns of gradient tracking, not training so need to track gradient, so pytorch keep tracking of less data
   y_preds = model_0(X_test)
   
# plot_predictions(predictions=y_preds)

#loss function

loss_fn = nn.L1Loss()

# print(loss_fn)

#optmiser

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01)

# print(optimizer)

epochs = 100
epoch_count = []
train_loss_values = []
test_loss_values = []


for epoch in range(epochs):
    model_0.train() # sets all parameters that require gradients to require gradients

    y_pred = model_0(X_train) # forward pass

    loss = loss_fn(y_pred, y_train) # calculate loss
    # print(f"Loss: {loss}")
    optimizer.zero_grad() # optmise the gradient

    loss.backward() # work out the gradient of the loss with respect to the parameters

    optimizer.step() #set the optimiser, would accumulate so need to zero after every loop

    model_0.eval() # turns off gradient tracking (more than just that), for when you are evaluating your model and don't need it 
    with torch.inference_mode():
        test_pred = model_0(X_test)

        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0: 
        epoch_count.append(epoch)
        train_loss_values.append(loss.detach().numpy())
        test_loss_values.append(test_loss.detach().numpy())

        # print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
        # print(model_0.state_dict())


# plt.plot(epoch_count, train_loss_values, label="Train loss")
# plt.plot(epoch_count, test_loss_values, label="Test loss")
# plt.title("Training and test loss curves")
# plt.ylabel("Loss")
# plt.xlabel("Epochs")
# plt.legend()
# plt.show()

#saving model
def save():
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = "worthflow_model.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    print(f"Saving model to: {MODEL_SAVE_PATH}")

    torch.save(obj=model_0.state_dict(),
            f=MODEL_SAVE_PATH)
    
    return
# save()

def load():
    loaded_model_0 = LinearRegressionModel()
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = "worthflow_model.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

    print(loaded_model_0.state_dict())
    print(model_0.state_dict())

    return 0

# load()





    





