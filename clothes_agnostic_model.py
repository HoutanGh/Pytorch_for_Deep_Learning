
import torch
from torch import nn
import torch.utils
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from helper_functions import print_train_time, accuracy_fn
from tqdm import tqdm
from torch import argmax

train_data = datasets.FashionMNIST(
    root="data", #where to download data to
    train=True, # test or train?
    download=True,
    transform=ToTensor(), # how we want to transform the data
    target_transform=None # how we want to transform the labels/targets
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

class_names = train_data.classes

BATCH_SIZE = 32

train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

train_features_batch, train_labels_batch = next(iter(train_dataloader))

device = "cuda" if torch.cuda.is_available() else "cpu"

class FashionModel_1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)


torch.manual_seed(42)
torch.cuda.manual_seed(42)

model_1 = FashionModel_1(input_shape=784, hidden_units=10, output_shape=len(class_names)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model_1.parameters(), lr=0.1)


def train_step(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimiser: torch.optim.Optimizer, 
               accuracy_fn, 
               device: torch.device = device):
    train_loss, train_acc = 0, 0

    model.train()

    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)

        train_loss += loss # for average loss per batch

        train_acc += accuracy_fn(y, y_pred.argmax(dim=1))

        optimiser.zero_grad()

        loss.backward()

        optimiser.step()

    train_loss /= len(train_dataloader)

    train_acc /= len(train_dataloader)

    

    print(f"Train Loss: {train_loss:.4f} | Train acc: {train_acc:.4f}%")

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)

            test_pred = model(X)

            test_loss += loss_fn(test_pred, y)

            test_acc += accuracy_fn(y, test_pred.argmax(dim=1))

        test_loss /= len(test_dataloader)

        test_acc /= len(test_dataloader)

    
    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}%")


torch.manual_seed(42)

train_time_start_on_gpu = timer()

epochs = 3

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch} \n-----------")
    train_step(model_1, train_dataloader, loss_fn, optimiser, accuracy_fn, device)

    test_step(model_1, test_dataloader, loss_fn, accuracy_fn, device)

train_time_end_gpu = timer()

total_train_time_model_1 = print_train_time(train_time_start_on_gpu, train_time_end_gpu, device)

#results dictionary

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.dataloader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device = device):
    loss, acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            y_pred = model(X)

            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y, y_pred.argmax(dim=1))


        loss /= len(data_loader)
        acc /= len(data_loader)

    
    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_acc": acc}


model_1_results = eval_model(model_1,
                             test_dataloader,
                             loss_fn, 
                             accuracy_fn)