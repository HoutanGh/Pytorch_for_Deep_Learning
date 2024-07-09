#convolutional neural network

import torch
from torch import nn
import torch.utils
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
from pathlib import Path

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

image, label = train_data[0]
class_names = train_data.classes

# print(image.shape, label)

# plt.imshow(image.squeeze(), cmap="gray")
# plt.show()

torch.manual_seed(42)
# fig = plt.figure(figsize=(9,9))
# rows , cols = 4, 4

# for i in range(1, rows*cols+1):
#     random_idx = torch.randint(0, len(train_data), size=[1]).item()
#     img, label = train_data[random_idx]
#     fig.add_subplot(rows, cols, i)
#     plt.imshow(img.squeeze(), cmap="gray")
#     plt.title(class_names[label])
#     plt.axis(False)

# plt.show()

# mini-batches

from torch.utils.data import DataLoader

BATCH_SIZE = 32

train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

train_features_batch, train_labels_batch = next(iter(train_dataloader))

# print(train_features_batch)


torch.manual_seed(42)

random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]

# plt.imshow(img.squeeze(), cmap="gray")
# plt.title(class_names[label])
# plt.axis(False)
# plt.show()

flatten_model = nn.Flatten()

x = train_features_batch[0]

output = flatten_model(x)

# print(output.shape)

class FashionModel(nn.Module):
    def __init__(self, input_shape=int, hidden_units=int, output_shape=int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)
    

torch.manual_seed(42)

model_0 = FashionModel(
    input_shape=784,
    hidden_units=10,
    output_shape=len(class_names)
).to("cpu")

dummy_x = torch.rand([1, 1, 28, 28])
# print(model_0(dummy_x))

from helper_functions import accuracy_fn

loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model_0.parameters(), lr=0.1)

# creating a function to time our experiment

from timeit import default_timer as timer

def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")

from tqdm.auto import tqdm
from torch import argmax

torch.manual_seed(42)

# train_time_start_on_cpu = timer()

epochs = 3

def clothes():
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n-------")

        train_loss = 0

        for batch, (X, y) in enumerate(train_dataloader):
            model_0.train()
            y_pred = model_0(X)

            loss = loss_fn(y_pred, y)

            train_loss += loss # for average loss per batch

            optimiser.zero_grad()

            loss.backward()

            optimiser.step()

        
            if batch % 400 == 0:
                print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples.")

        train_loss /= len(train_dataloader)

        test_loss, test_acc = 0, 0
        model_0.eval()
        with torch.inference_mode():
            for X_test, y_test in test_dataloader:
                test_pred = model_0(X_test)

                test_loss += loss_fn(test_pred, y_test)

                test_acc += accuracy_fn(y_test, test_pred.argmax(dim=1))

            test_loss /= len(test_dataloader)

            test_acc /= len(test_dataloader)

        
        print(f"\nTrain Loss: {train_loss:.4f} | Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

    return 0


# train_time_end_on_cpu = timer()
# total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu, 
#                                            end=train_time_end_on_cpu,
#                                            device=str(next(model_0.parameters()).device))

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.dataloader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device):
    loss, acc = 0, 0

    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y, y_pred.argmax(dim=1))


        loss /= len(data_loader)
        acc /= len(data_loader)

    
    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_acc": acc}
    
# model_0_results = eval_model(model_0, test_dataloader, loss_fn, accuracy_fn)

# print(model_0_results)

#agnostic code


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

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)

        train_loss += loss # for average loss per batch

        train_acc += accuracy_fn(y, y_pred.argmax(dim=1))

        optimiser.zero_grad()

        loss.backward()

        optimiser.step()

    
        # if batch % 400 == 0:
        #     print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples.")

    train_loss /= len(train_dataloader)

    train_acc /= len(train_dataloader)

    

    print(f"Train Loss: {train_loss:.4f} | Train acc: {train_acc:.4f}%")

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              optimiser: torch.optim.Optimizer,
              accuracy_fn,
              device: torch.device = device):
    
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            test_pred = model(X)

            test_loss += loss_fn(test_pred, y)

            test_acc += accuracy_fn(y, test_pred.argmax(dim=1))

        test_loss /= len(test_dataloader)

        test_acc /= len(test_dataloader)

    
    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}%")


torch.manual_seed(42)

# train_time_start_on_gpu = timer()

epochs = 3

# for epoch in tqdm(range(epochs)):
#     print(f"Epoch: {epoch} \n-----------")
#     train_step(model_1, train_dataloader, loss_fn, optimiser, accuracy_fn, device)

#     test_step(model_1, test_dataloader, loss_fn, accuracy_fn, device)

# train_time_end_gpu = timer()

# total_train_time_model_1 = print_train_time(train_time_start_on_gpu, train_time_end_gpu, device)


#convolutional neural network

class FashionModel_2(nn.Module):
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



    # torch.manual_seed(42)

    # model_2 = FashionModel_2(input_shape=1, hidden_units=10, output_shape=len(class_names)).to(device)

    #testing with dummy data
    #conv layer

    # images = torch.randn(size=(32, 3, 64, 64))
    # test_image = images[0]

    # conv_layer = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=0)

    # conv_output = conv_layer(test_image.unsqueeze(0))

    # max_pool_layer = nn.MaxPool2d(kernel_size=2)

    # test_image_after_conv = conv_layer(test_image.unsqueeze(dim=0))

    # test_image_after_maxpool = max_pool_layer(test_image_after_conv)


    # print(f"original image shape: {test_image.shape}")
    # print(f"unsqueezed test image: {test_image.unsqueeze(dim=0).shape}")

    # print(f"shape after convo layer: {test_image_after_conv.shape}")

    # print(f"after maxpool: {test_image_after_maxpool.shape}")

    #maxpool2d

    # random_tensor = torch.rand((1, 1, 2, 2))

    # max_pool_layer = nn.MaxPool2d(kernel_size=2)

    # max_pool_tensor = max_pool_layer(random_tensor)

    # print(random_tensor)
    # print(random_tensor.shape)
    # print(max_pool_tensor)
    # print(max_pool_tensor.shape)

torch.manual_seed(42)

model_2 = FashionModel_2(input_shape=1, hidden_units=10, output_shape=len(class_names)).to(device)

# rand_image_tensor = torch.randn((1, 28, 28))

# print(model_2(rand_image_tensor.unsqueeze(0).to(device)))



loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model_2.parameters(), lr=0.1)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

start_time_model_2 = timer()

epochs = 5

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch} \n--------")
    train_step(model_2, train_dataloader, loss_fn, optimiser, accuracy_fn, device)
    test_step(model_2, test_dataloader, loss_fn, optimiser, accuracy_fn, device)

end_time_model_2 = timer()

total_time_model_2 = print_train_time(start_time_model_2, end_time_model_2, device)



#make and evaluate the best models

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

def loading_model():
    loaded_model = FashionModel_2(input_shape=1, hidden_units=10,output_shape=len(class_names)).to(device)
    loaded_model.load("cnn_model.pth", "models", device)

    import random
    random.seed(42)
    test_samples = []
    test_labels = []

    for sample, label in random.sample(list(test_data), k=9):
        test_samples.append(sample)
        test_labels.append(label)

    # plt.imshow(test_samples[1].squeeze(), cmap="gray")
    # plt.title(class_names[test_labels[1]])
    # plt.show()

    #make predictions

    pred_probs = make_predictions(loaded_model, data=test_samples)

    # print(f"Test sample image shape: {test_samples[0].shape}\nTest sample label: {test_labels[0]} ({class_names[test_labels[0]]})")


    # print(test_samples)
    # print(test_labels)
    # print(pred_probs[:2])

    pred_classes = pred_probs.argmax(dim=1)

    # print(pred_classes)
    # print(test_labels)

    #have to run the model, maybe could use load_model then run the pred labels to check

    plt.figure(figsize=(9,9))
    nrows = 3
    ncols = 3

    for i, sample in enumerate(test_samples):
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(sample.squeeze(), cmap="gray")
        pred_label = class_names[pred_classes[i]]
        truth_label = class_names[test_labels[i]]
        title_text = f"Pred: {pred_label} | Truth: {truth_label}"

        if pred_label == truth_label:
            plt.title(title_text, fontsize=10, c="g")
        else:
            plt.title(title_text, fontsize=10, c="r")

# plt.show()

#confusion matrix

from tqdm.auto import tqdm

y_preds = []

model_2.eval()
with torch.inference_mode():
    for X, y in tqdm(test_dataloader, desc="Making predictions..."):
        X, y = X.to(device), y.to(device)

        y_logit = model_2(X)

        y_pred = torch.softmax(y_logit.squeeze(), dim=0).argmax(dim=1)

        y_preds.append(y_pred.cpu())

# print(y_preds[:1])

y_preds_tensor = torch.cat(y_preds)

# print(y_preds_tensor[:10])

import mlxtend

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')

confmat_tensor = confmat(preds=y_preds_tensor, target=test_data.targets)

fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names=class_names,
    figsize=(10,7)
)
plt.show()
















    





