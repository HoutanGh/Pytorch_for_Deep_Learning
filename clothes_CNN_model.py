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

# doing 2D images so 2D Convolutional Layers are needed 

class FashionModel(nn.Module):
    def __init__(self, input_shape, hidden_units: int, output_shape: int):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,      # input_shape = 1 for grayscale images and 3 for RGB images
                      out_channels=hidden_units,
                      kernel_size=3,                # convolutional kernel filter - basically how much of the 
                                                    # picture it captues at a given time, 3x3 generally enough 
                                                    # to detect fine details but large enough to capture important patterns
                      stride=1,                     # the number of pixels by which the filter moves across the input image
                      padding=1),                   # one pixel is added around the border of image
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)             # reduces the spatial dims of input, 2x2 kernel means input divided into
                                                    # 2x2 regions and max value from each region is taken
                                                    # reduces spatial dims by factor of 2
                
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

        # for forward pass, fully connected linear layers expect 1D input (batch_size, features) but the output 
        # from the convolutional and pooling layers is typically a 3D tensor (batch_size, channels, height, width)
        self.classifier = nn.Sequential(
            nn.Flatten(),                               # flattneing the 3D tensor into 1D
            nn.Linear(in_features=hidden_units * 7 * 7, # linear transformation on the input data
                                                        # 7 * 7 is the dimensions by the pooling layer
                                                        # since the input image is 28x28 after two 2x2 max pooling
                                                        # layers, the spatial dimensions are reduced by a factor of 4

                      out_features=output_shape)        # output shape is 10, representing the 10 different clothing items

        )

    def forward(self, x):
        x = self.conv_block_1(x)
        # print(f"Output shape after conv_block_1: {x.shape}")
        x = self.conv_block_2(x)
        # print(f"Output shape after conv_block_2: {x.shape}")
        x = self.classifier(x)
        # print(f"Output shape after classifier: {x.shape}")
        return x
    
    
    # incoporated the save and load functions into to class for convenience
    def save(self, model_name: str, model_path: str) -> torch.nn.Module:
        model_save_path = Path(model_path) / model_name
        print(f"Saving model to: {model_save_path}")
        torch.save(self.state_dict(), model_save_path)

    def load(self, model_name: str, model_path: str, device: torch.device) -> torch.nn.Module:
        model_save_path = Path(model_path) / model_name
        print(f"Loading model from: {model_save_path}")
        self.load_state_dict(torch.load(model_save_path, map_location=device))
        self.to(device)

# FashionMNIST - 60,000 training examples and 10,000 test examples, each of size 28x28 grayscale images in 10 categories
train_data = datasets.FashionMNIST( 
    root="data", # where to download data to
    train=True, # test or train?
    download=True, # download from internet if not already present 
    transform=ToTensor(), # converts PIL Image or Numpy ndarray into Pytorch tensor
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

BATCH_SIZE = 32 # number of samples that will be processed in a single batch

# DataLoader provides an iterable over a dataset with support for batching, shuffling and parallel data loading
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False) # don't need to shuffle test data


# used to inspect a single batch of data to ensure that the data loading process is working correctly
# train_features_batch, train_labels_batch = next(iter(train_dataloader))

device = "cuda" if torch.cuda.is_available() else "cpu"




    
model = FashionModel(input_shape=1, hidden_units=10, output_shape=len(class_names)).to(device)

# loss function that combines nn.LogSoftmax and nn.NLLLoss in one single class
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.1)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

start_time_model = timer()

epochs = 3

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch} \n--------")
    # functions from my_helper_functions.py that follow similar pattern to other models
    train_step(model, train_dataloader, loss_fn, optimiser, accuracy_fn, device)
    test_step(model, test_dataloader, loss_fn, optimiser, accuracy_fn, device)

end_time_model = timer()

total_time_model = print_train_time(start_time_model, end_time_model, device)

model_results = eval_model(model, test_dataloader, loss_fn, accuracy_fn, device)

# model.save()

def make_predictions(model: torch.nn.Module,
                    data: list,
                    device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device) # add extra dim to tensor at dim=0, since the 
                                                               # model expects the input to have a batch dim

            pred_logit = model(sample)

            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0) # need to remove dim for softmax

            pred_probs.append(pred_prob.cpu()) # .cpu when saving

    return torch.stack(pred_probs) # stacks the list of prediction probability tensors into a single tensor

loaded_model = FashionModel(input_shape=1, hidden_units=10, output_shape=len(class_names)).to(device)
loaded_model.load("cnn_model.pth", "models", device)
