import torch
from torch import nn
import matplotlib.pyplot as plt
import requests
from zipfile import ZipFile
from pathlib import Path
import os
import random
from PIL import Image
import numpy as np
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import Tuple, Dict, List

# Disable cuDNN benchmarking
# lets the cuDNN library select the best algorithms for the hardware
torch.backends.cudnn.benchmark = True

# Disable cuDNN (optional, if other solutions don't work)
# torch.backends.cudnn.enabled = False
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = Path("data/")
    image_path = data_path / "pizza_steak_sushi"

    # if image directory doesnt exist then create directory and get the zip file containing the image data
    if not image_path.is_dir():
        image_path.mkdir(parents=True, exist_ok=True)
        with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
            request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
            f.write(request.content)
        
        # get content from zip file
        with ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
            zip_ref.extractall(image_path)

    def walk_through_dir(dir_path):
        for dirpath, dirnames, filenames in os.walk(dir_path):
            print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

    # walk_through_dir(image_path)

    train_dir = image_path / "train"
    test_dir = image_path / "test"

    # code to be able to do multiple transformations, reshape and change to tensor
    # data_transform = transforms.Compose([
    #     transforms.Resize(size=(64, 64)),
    #     transforms.RandomHorizontalFlip(0.5),
    #     transforms.ToTensor()
    # ])

    # train_data = datasets.ImageFolder(root=train_dir, transform=data_transform)
    # test_data = datasets.ImageFolder(root=test_dir, transform=data_transform)

    

    # NUM_WORKERS = os.cpu_count()

    # train_dataloader = DataLoader(dataset=train_data, batch_size=32, num_workers=NUM_WORKERS, shuffle=True)
    # test_dataloader = DataLoader(dataset=test_data, batch_size=32, num_workers=NUM_WORKERS, shuffle=False)

    class TinyVGG(nn.Module):
        def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
            super().__init__()
            self.conv_block_1 = nn.Sequential(
                nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.conv_block_2 = nn.Sequential(
                nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            # size of images are 64x64 and go through two maxpool layers so are reduced to 16x16
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(in_features=hidden_units * 16 * 16, out_features=output_shape))

        def forward(self, x):
            x = self.conv_block_1(x)
            x = self.conv_block_2(x)
            x = self.classifier(x)
            return x
        
        def save(self, model_name: str, model_path: str) -> torch.nn.Module:
            model_save_path = Path(model_path) / model_name
            print(f"Saving to {model_save_path}")
            torch.save(self.state_dict(), model_save_path)

        def load(self, model_name: str, model_path: str, device: torch.device) -> torch.nn.Module:
            model_save_path = Path(model_path) / model_name
            print(f"Loading model from: {model_save_path}")
            self.load_state_dict(torch.load(model_save_path, map_location=device))
            self.to(device)




    from torchvision import datasets
    
    simple_transform = transforms.Compose([ 
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
        
    
    train_data_simple = datasets.ImageFolder(root=train_dir, transform=simple_transform)
    test_data_simple = datasets.ImageFolder(root=test_dir, transform=simple_transform)

    class_names = train_data_simple.classes

    # Turn data into DataLoaders
    import os
    from torch.utils.data import DataLoader

    # Setup batch size and number of workers 
    BATCH_SIZE = 32
    NUM_WORKERS = 1 # number of CPU cores I think?
    print(f"Creating DataLoader's with batch size {BATCH_SIZE} and {NUM_WORKERS} workers.")

    # Create DataLoader's
    train_dataloader_simple = DataLoader(train_data_simple, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True, 
                                        num_workers=NUM_WORKERS)

    test_dataloader_simple = DataLoader(test_data_simple, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=False, 
                                        num_workers=NUM_WORKERS)

    torch.manual_seed(42)

    # basically just trying to see how bad the model is right now by seeing the raw output of the model on a single image
    model_0 = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(train_data_simple.classes)).to(device)

    # retrieves next batch of images and labels from the training DataLoader
    img_batch, label_batch = next(iter(train_dataloader_simple))
    # extracting a single image and label
    # unsqueeze necessary because the model still expects a batch of images even if it is 1
    img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
    print(f"Single image shape: {img_single.shape}\n")

    # setting model to evaluation mode
    model_0.eval()
    # disabling gradient calculation
    with torch.inference_mode():
        pred = model_0(img_single.to(device))

    print(f"Output logits:\n{pred}\n")
    print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
    print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
    print(f"Actual label:\n{label_single}")

    # easier way to get info
    from torchinfo import summary
    summary(model_0, input_size=[1,3,64,64])

    def train_step(model: torch.nn.Module, dataloader:torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimiser: torch.optim.Optimizer):
        model.train()

        train_loss, train_acc = 0, 0

        # dataloader provides batches of data, batch is the index of current batch, (X, y) are the batch of inputs and corresponding labels
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            y_pred_logits = model(X)

            loss = loss_fn(y_pred_logits, y)

            # adds the loss of the current batch to the total training loss
            train_loss += loss.item()

            optimiser.zero_grad()

            loss.backward()

            optimiser.step()

            # torch.softmax converts logits to probabilities
            # torch.argmax gets the class with the highest probability
            y_pred_class = torch.argmax(torch.softmax(y_pred_logits, dim=1), dim=1)

            # adds accuracy of current batch to total training accuracy
            train_acc += (y_pred_class == y).sum().item()/len(y_pred_logits)

        train_loss = train_loss / len(dataloader)

        train_acc = train_acc / len(dataloader)

        return train_loss, train_acc


    def test_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module):
        model.eval()

        test_loss, test_acc = 0, 0

        with torch.inference_mode():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)

                test_pred_logits = model(X)

                loss = loss_fn(test_pred_logits, y)
                test_loss += loss.item()

                # softmax does not seem to be a requirement
                test_pred_labels = torch.argmax(test_pred_logits, dim=1)
                test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)

        return test_loss, test_acc
    
    from tqdm.auto import tqdm

    def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader,
              optimiser: torch.optim.Optimizer, loss_fn: torch.nn.Module, epochs = 5):
        
        # dictionary for storing data for each epoch
        results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimiser=optimiser)
            test_loss, test_acc = test_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn)

            print(f"Epoch: {epoch+1} | "
                  f"train_loss: {train_loss:.4f} | "
                  f"train_acc: {train_acc:.4f} | "
                  f"test_loss: {test_loss:.4f} | "
                  f"test_acc: {test_acc:.4f}")

            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

        return results
    
    # seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Number of times we are running the training and testing functions
    NUM_EPOCHS = 5

    # Initialise model
    model_0 = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(class_names)).to(device)

    # Defining loss function and optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

    # Want to see how long the model takes
    from timeit import default_timer as timer
    start_timer = timer()

    model_0_results = train(model=model_0, 
                            train_dataloader=train_dataloader_simple, 
                            test_dataloader=test_dataloader_simple, 
                            loss_fn=loss_fn, 
                            optimiser=optimiser, 
                            epochs = NUM_EPOCHS)
    
    end_timer = timer()

    # Visualising the model
    print(f"Total training time: {end_timer-start_timer:.3f} seconds")
    print(model_0_results)
    print(len(model_0_results["train_loss"]))
    
    # Visualise the training and validation loss and accuracy over the epochs
    # Function takes in a dictionary where the keys are strings and the values are lists of floats
    def plot_loss_curves(results: Dict[str, List[float]]):
        
        epochs = range(len(results["train_loss"]))

        # Extracting the data, Note: function works specifically for our dictionary    
        loss = results["train_loss"]
        test_loss = results["test_loss"]
        accuracy = results["train_acc"]
        test_acc = results["test_acc"]

        plt.figure(figsize=(15, 7))       
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, label='train_loss')
        plt.plot(epochs, test_loss, label='test_loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracy, label='train_acc')
        plt.plot(epochs, test_acc, label='test_acc')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        
        plt.show()

    # plot_loss_curves(model_0_results)


    # Making prediction on a new image sample
    import requests

    custom_image_path = data_path / "04-pizza-dad.jpeg"


    if not custom_image_path.is_file():
        # Opens file to write contents in
        with open (custom_image_path, "wb") as f:
            request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
            print(f"Downloading {custom_image_path}")
            f.write(request.content)

    else:
        print(f"{custom_image_path} already exists")


    import torchvision

    # custom_image_uint8 = torchvision.io.read_image(str(custom_image_path))
    # print(f"Custom image tensor:\n{custom_image_uint8}\n")
    # print(f"Custom image shape: {custom_image_uint8.shape}\n")
    # print(f"Custom image dtype: {custom_image_uint8.dtype}")

    # model_0.eval()
    # with torch.inference_mode():
    #     model_0(custom_image_uint8.to(device)) #but this is not in correct datatype (float32)
    
    # torchvision.io.read_image reads the image from the file and returns it as a tensor
    custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)
    print(f"Custom image tensor:\n{custom_image}\n")
    print(f"Custom image shape: {custom_image.shape}\n")
    print(f"Custom image dtype: {custom_image.dtype}")

    plt.imshow(custom_image.permute(1, 2, 0))
    plt.title(f"Image shape: {custom_image.shape}")
    plt.axis(False)

    # Defining the transform and the transformed image
    custom_image_transform = transforms.Compose([transforms.Resize((64, 64)),])
    custom_image_transformed = custom_image_transform(custom_image)

    print(f"Original shape: {custom_image.shape}")
    print(f"New shape: {custom_image_transformed.shape}")


    # still need a dimension for batch size so use unsqueeze

    model_0.eval()
    with torch.inference_mode():
        # Retrieving the model's raw output logits for the image
        custom_image_pred = model_0(custom_image_transformed.unsqueeze(dim=0).to(device))

    
    print(f"Prediction logits: {custom_image_pred}")
    custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
    print(f"Predicition probabilities: {custom_image_pred_probs}")
    custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
    print(f"Prediction label in index form: {custom_image_pred_label}")
    custom_image_pred_class = class_names[custom_image_pred_label.cpu()]
    print(f"Actual label: {custom_image_pred_class}")


    # Making predictions on a new image and visualising the results
    def pred_and_plot(model:torch.nn.Module,
                      image_path: str,
                      class_names: List[str] = None,
                      transform=None,
                      device: torch.device = device):
        
        target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)
        # Normalising the image to scale the pixel values to the range [0,1]
        target_image = target_image / 255

        # If there is a transform, apply it
        if transform:
            target_image = transform(target_image)

        model.to(device)

        model.eval()
        with torch.inference_mode():
            target_image = target_image.unsqueeze(dim=0)
            target_image_pred = model(target_image.to(device))
        
        # Softmax for prob, Argmax for getting label
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
        target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

        plt.imshow(target_image.squeeze().permute(1, 2, 0))

        # Giving class name if exists or just the label if not
        if class_names:
            title = f"Pred: {class_names[custom_image_pred_label.cpu()]} | Probs: {target_image_pred_probs.max().cpu():.4f}"
        else:
            title = f"Pred: {target_image_pred_label} | Probs: {target_image_pred_probs.max().cpu():.4f}"

        plt.title(title)
        plt.axis(False)
        plt.show()
    
    pred_and_plot(model=model_0, image_path=custom_image_path, class_names=class_names, transform=custom_image_transform, device=device)






            

