#custom datasets

import torch
from torch import nn
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

import requests
import zipfile
from pathlib import Path

data_path = Path("data/")

image_path = data_path / "pizza_steak_sushi"

# print(image_path)

# if image_path.is_dir():
#     print(f"{image_path} directory already exists") 
# else:
#     print(f"{image_path} des not exist, creating one...")
#     image_path.mkdir(parents=True, exist_ok=True)

# with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
#     request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
#     print("Downloading...")
#     f.write(request.content)

# with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
#     print("Unzipping pizza, steak and sushi data...")
#     zip_ref.extractall(image_path)


import os

def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

# print(walk_through_dir(image_path))

train_dir = image_path / "train"
test_dir = image_path / "test"

# print(train_dir, test_dir)

import random
from PIL import Image

random.seed(42)

image_path_list = list(image_path.glob("*/*/*.jpg")) #each / is a directory and the .jpg is for the images

# print(image_path_list)

random.seed(42)

random_image_path = random.choice(image_path_list)

image_class = random_image_path.parent.stem

# print(random_image_path)

# print(image_class)

img = Image.open(random_image_path)

# print(f"Random image path: {random_image_path}")
# print(f"Image class: {image_class}")
# print(f"Image height: {img.height}")
# print(f"Image width: {img.height}")

# plt.imshow(img)
# plt.axis('off')
# plt.show()

import numpy as np

# img_as_array = np.asarray(img)
# plt.figure(figsize=(10, 7))
# plt.imshow(img_as_array)
# plt.axis(False)
# plt.title(f"Image class: {image_class} | Image Shape: {img_as_array.shape} --> [height, width, colour channels]")
# plt.show()

import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor()
])

# print(data_transform(img))

def plot_transformed_images(image_paths, transform, n=3, seed=42):
    random.seed(42)

    random_image_path = random.sample(image_paths, k=n)
    for image_path in random_image_path:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            transformed_image = transform(f).permute(1, 2, 0) # permutation to match the [C, H, W] for the tensors
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Tranformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

# plot_transformed_images(image_path_list, transform=data_transform, n=3)

# plt.show()

from torchvision import datasets
train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform,
                                  target_transform=None)

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)

# print(f"Train data:\n{train_data}\nTest data:\n{test_data}")


class_names = train_data.classes
class_dic = train_data.class_to_idx

# print(class_dic)

# print(train_data.samples[0])

# print(train_data[0])

img, label = train_data[0][0], train_data[0][1]

def print_img_info():
            
    print(f"Image tensor:\n{img}")
    print(f"Image shape: {img.shape}")
    print(f"Image datatype: {img.dtype}")
    print(f"Image label: {label}")
    print(f"Label datatype: {type(label)}")

img_permute = img.permute(1, 2, 0)

def plot_permute_image():
    plt.figure(figsize=(10,7))
    plt.imshow(img_permute)
    plt.axis(False)
    plt.title(class_names[label], fontsize=14)

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=1,
                              num_workers=6, #how cpu cores used to load data
                              shuffle=True)


test_dataloader = DataLoader(dataset=test_data, batch_size=1, num_workers=6, shuffle=False)


img, label = next(iter(train_dataloader))

# print(f"Image shape: {img.shape}")
# print(f"Label shape: {label.shape}")


# have used existing ImageFolder data loading class
# want to develop our own for data that doesn have a pre-built function

#Loading Image Data with a Custom Dataset

from torch.utils.data import Dataset
from typing import Tuple, Dict, List

#helper function for class names

target_directory = train_dir

# print(list(os.scandir(target_directory)))

class_names_found = sorted([entry.name for entry in list(os.scandir(target_directory))])

# print(class_names_found)

def find_classes(directory:str) -> Tuple[List[str], Dict[str, int]]:
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    if not classes:
        raise FileNotFoundError(f"Couldn/t find any classes in {directory}... please check file structure.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    return classes, class_to_idx

find_classes(target_directory)

#custom dataset that replicates ImageFolder
from torch.utils.data import Dataset

class ImageFolderCustom(Dataset):
    def __init__(self, targ_dir: str, transform=None) -> None:
        super().__init__()
        self.paths = list(Path(targ_dir).glob("*/*.jpg"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)

    def load_images(self, index:int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, index:int) -> Tuple[torch.tensor, int]:
        img = self.load_images(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx
            

if __name__ == "__main__":            
    train_transforms = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor()
    ])

    train_data_custom = ImageFolderCustom(targ_dir=train_dir,
                                          transform=train_transforms)
    
    test_data_custom = ImageFolderCustom(targ_dir=test_dir, transform=test_transforms)

    # print(train_data_custom.classes)

    def display_random_images(dataset: torch.utils.data.Dataset,
                              classes: List[str] = None,
                              n: int = 10,
                              display_shape: bool = True,
                              seed: int = None):
        if n > 10:
            n = 10
            display_shape = False
            print(f"n shouldn't be larger than 10 for display purposes")
        
        if seed:
            random.seed(42)

        random_samples_idx = random.sample(range(len(dataset)), k=n)

        plt.figure(figsize=(16, 8))

        for i, targ_sample in enumerate(random_samples_idx):
            targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]
            targ_image_adjust = targ_image.permute(1, 2, 0)

            plt.subplot(1, n, i+1)
            plt.imshow(targ_image_adjust)
            plt.axis(False)
            if classes:
                title = f"class: {classes[targ_label]}"
                if display_shape:
                    title = title + f"\nshape: {targ_image_adjust.shape}"
            
            plt.title(title)
        plt.show()


    # display_random_images(train_data, n=5, classes=class_names, seed=42)

    BATCH_SIZE = 32
    train_dataloader_custom = DataLoader(dataset=train_data_custom,
                                         batch_size=BATCH_SIZE,
                                         num_workers=0,
                                         shuffle=True)
    
    test_dataloader_custom = DataLoader(dataset=test_data_custom, # use custom created test Dataset
                                        batch_size=1, 
                                        num_workers=0, 
                                        shuffle=False)
    
    img_custom, label_custom = next(iter(train_dataloader_custom))

    #data augmentation - artificially adding diversity to your data

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor()
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image_path_list = list(image_path.glob("*/*/*.jpg"))

    # plot_transformed_images(image_path_list, transform=train_transforms, n=3, seed=42)

    # plt.show()

    #do model without augmentation

    simple_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_data_simple = datasets.ImageFolder(root=train_dir, transform=simple_transform)
    test_data_simple = datasets.ImageFolder(root=test_dir, transform=simple_transform)

    NUM_WORKERS = os.cpu_count()

    train_dataloader_simple = DataLoader(train_data_simple, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_dataloader_simple = DataLoader(test_data_simple, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    class TinyVGG(nn.Module):
        def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
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
                nn.MaxPool2d(kernel_size=2,
                             stride=2)
            )
            self.conv_block_2 = nn.Sequential(
                nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(in_features=hidden_units*16*16, out_features=output_shape))

        
        def forward(self, x):
            x = self.conv_block_1(x)
            x = self.conv_block_2(x)
            x = self.classifier(x)

            return x
        

    torch.manual_seed(42)
    model_0 = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(train_data.classes)).to(device)

    # print(model_0)

    #forward pass on a single image

    img_batch, label_batch = next(iter(train_dataloader_simple))

    print(img_batch.shape, label_batch.shape)

    # print(model_0(img_batch.to(device)))

    img_single, label_single = img_batch[0].unsqueeze(dim=1), label_batch[0]

    print(f"Single image shape: {img_single.shape}\n")

    model_0.eval()
    with torch.inference_mode():
        pred = model_0(img_single.to(device))
    
    print(f"Output logits:\n{pred}\n")
    print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
    print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
    print(f"Actual label:\n{label_single}")










            





    

















