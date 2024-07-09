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
from my_helper_functions import train_step, test_step, eval_model
from pathlib import Path

class FashionModel(nn.Module):
    def __init__(self, input_shape, hidden_units: int, output_shape: int):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
                
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)                 
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7 * 7,
                      out_features=output_shape)

        )

    def forward(self, x):
        x = self.conv_block_1(x)
        # print(f"Output shape after conv_block_1: {x.shape}")
        x = self.conv_block_2(x)
        # print(f"Output shape after conv_block_2: {x.shape}")
        x = self.classifier(x)
        # print(f"Output shape after classifier: {x.shape}")
        return x
    
    def save(self, model_name: str, model_path: str) -> torch.nn.Module:
        model_save_path = Path(model_path) / model_name
        print(f"Saving model to: {model_save_path}")
        torch.save(self.state_dict(), model_save_path)

    def load(self, model_name: str, model_path: str, device: torch.device) -> torch.nn.Module:
        model_save_path = Path(model_path) / model_name
        print(f"Loading model from: {model_save_path}")
        self.load_state_dict(torch.load(model_save_path, map_location=device))
        self.to(device)

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




    
model = FashionModel(input_shape=1, hidden_units=10, output_shape=len(class_names)).to(device)

# loss_fn = nn.CrossEntropyLoss()
# optimiser = torch.optim.SGD(model.parameters(), lr=0.1)

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

# start_time_model = timer()

# epochs = 3

# for epoch in tqdm(range(epochs)):
#     print(f"Epoch: {epoch} \n--------")
#     train_step(model, train_dataloader, loss_fn, optimiser, accuracy_fn, device)
#     test_step(model, test_dataloader, loss_fn, optimiser, accuracy_fn, device)

# end_time_model = timer()

# total_time_model = print_train_time(start_time_model, end_time_model, device)

# model_results = eval_model(model, test_dataloader, loss_fn, accuracy_fn, device)



# def save():
#     MODEL_PATH = Path("models")
#     MODEL_NAME = "cnn_model.pth"
#     MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

#     print(f"Saving model to: {MODEL_SAVE_PATH}")

#     torch.save(model.state_dict(), MODEL_SAVE_PATH)


# # save()

# def load():
#     MODEL_PATH = Path("models")
#     MODEL_NAME = "cnn_model.pth"
#     MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

#     loaded_model = FashionModel(input_shape=1, hidden_units=10, output_shape=len(class_names)).to(device)

#     loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH))

#     return loaded_model



def make_predictions(model: torch.nn.Module,
                    data: list,
                    device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)

            pred_logit = model(sample)

            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            pred_probs.append(pred_prob.cpu())

    return torch.stack(pred_probs)

loaded_model = FashionModel(input_shape=1, hidden_units=10, output_shape=len(class_names)).to(device)
loaded_model.load("cnn_model.pth", "models", device)
