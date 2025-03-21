from PIL import Image
import torch
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np
import os


# This file contains code to generate and save pytorch
# tensors out of data we aim to use to train in network in CNN.py
# We merge two data sets here.


# The classes in the classification model we aim to train
characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
              'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z']

# The resolutions of the images in the two datasets are different
# We will resize them all so that they have shape 30x40
image_width = 30
image_height = 40


# Prepare tensors to receive the data
X_tensor = torch.empty(0, 1, image_height, image_width)
Y_tensor = torch.empty(0, dtype = torch.int64)

# We begin with the first dataset
print("Loading first dataset")
for i in range(len(characters)):
    path = "./datasets/license_plate_characters_dataset/" + characters[i]
    print("loading character " + characters[i])
    for file in os.listdir(path):
        # path and file_path are strings containing filepaths navigating through the dataset
        file_path = path + "/" + os.fsdecode(file)
        # We use PIL to open a .jpg file, resize it, and torchvision.transforms to turn it into a torch tensor
        PIL_image = Image.open(file_path)
        PIL_image = PIL_image.resize((image_width, image_height))
        img_tensor = transforms.PILToTensor()(PIL_image)
        # We store all the images into a big tensor X_tensor
        X_tensor = torch.cat((X_tensor, img_tensor.unsqueeze(0)), dim=0)
        # We keep track of the target in a tensor Y_tensor
        Y_tensor = torch.cat((Y_tensor, torch.tensor([i], dtype = torch.int64)))

# Sanity check regarding the shape
print(X_tensor.shape) #prints "torch.Size([35500, 1, 40, 30])": 35500 images, 1 grayscale channel, 30x40 pixels
print(Y_tensor.shape) #prints "torch.Size([35500])"

# We continue with the second dataset
print("Loading second dataset")
for i in range(len(characters)):
    path = "./datasets/general_characters_dataset/" + characters[i]
    print("loading character " + characters[i])
    for file in os.listdir(path):
        # path and file_path are strings containing filepaths navigating through the dataset
        file_path = path + "/" + os.fsdecode(file)
        # We use PIL to open a .jpg file, resize it, and torchvision.transforms to turn it into a torch tensor
        PIL_image = Image.open(file_path)
        PIL_image = PIL_image.resize((image_width, image_height))
        img_tensor = transforms.PILToTensor()(PIL_image)
        # We store all the images into a big tensor X_tensor
        X_tensor = torch.cat((X_tensor, img_tensor.unsqueeze(0)), dim=0)
        # We keep track of the target in a tensor Y_tensor
        Y_tensor = torch.cat((Y_tensor, torch.tensor([i], dtype = torch.int64)))

# Sanity check regarding the shape
print(X_tensor.shape) #prints "torch.Size([56535, 1, 40, 30])": 35500 images, 1 grayscale channel, 30x40 pixels
print(Y_tensor.shape) #prints "torch.Size([56535])"

# Save the tensors
torch.save(X_tensor, "./pytorch_files/X_tensor.pt")
torch.save(Y_tensor, "./pytorch_files/Y_tensor.pt")

# Display 25 images to verify the conversion
fig, axes = plt.subplots(5, 5, figsize=(10,10))
rng = np.random.default_rng(4)
indices = rng.choice(np.arange(len(X_tensor)), 25, replace=False).reshape((5,5))
for i in range(5):
    for j in range(5):
        idx = indices[i,j]
        axes[i,j].imshow(X_tensor[idx][0], interpolation=None, cmap='gray', vmin=0, vmax=255)
        axes[i,j].set_xticks([])
        axes[i,j].set_yticks([])
plt.show()
plt.close()
