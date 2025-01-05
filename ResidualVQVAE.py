###################################
#########   Imports  ##############
###################################

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import ssl
from torchvision import datasets
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import time
from sklearn.cluster import KMeans

import tarfile
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import pickle
import matplotlib.pyplot as plt
import random
import math

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:128'


import utilis
import logging

del KMeans

BATCH_SIZE = 16 # Changed from 16
LEARNING_RATE = 1e-5
EPOCHS = 1
ARCH = 'CIFAR'
SIZE = 64  # 64 IS FOR IMAGEWOOF
# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
num_workers = 0

###################################################################################################
###################################################################################################
#################################         Data Arrangment         #################################
###################################################################################################
###################################################################################################

if ARCH == 'CIFAR':
    NUM_CLASSES = 100
    FEATURES = 81920


    def get_test_transforms():
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        return test_transform


    def get_train_transforms():
        transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        return transform


    train_transform = get_train_transforms()
    test_transform = get_test_transforms()

    ssl._create_default_https_context = ssl._create_unverified_context
    path = "/tmp/cifar100"
    trainset = datasets.CIFAR100(root=path, train=True, download=True, transform=train_transform)
    testset = datasets.CIFAR100(root=path, train=False, download=True, transform=test_transform)

    # Define the size of the subset
    train_subset_size = int(len(trainset) * 0.01)
    test_subset_size = int(len(testset) * 0.01)

    # Create random indices for the subset
    train_indices = np.random.choice(len(trainset), train_subset_size, replace=False)
    test_indices = np.random.choice(len(testset), test_subset_size, replace=False)

    # Create the subset
    train_subset = Subset(trainset, train_indices)
    test_subset = Subset(testset, test_indices)

    # Create
    #     BATCH_SIZE = 64 DataLoaders for the subsets

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

elif ARCH == 'IMAGEWOOF':
    NUM_CLASSES = 10
    FEATURES = 327680
    ssl._create_default_https_context = ssl._create_unverified_context
    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz"
    download_url(dataset_url, '.')

    with tarfile.open('./imagewoof2-160.tgz', 'r:gz') as tar:  # read file in r mode
        tar.extractall(path='./data')  # extract all folders from zip file and store under folder named data

    data_dir = './data/imagewoof2-160'
    # print(os.listdir(data_dir))
    # print(os.listdir('./data/imagewoof2-160/train'))
    # print(len(os.listdir('./data/imagewoof2-160/train')))
    classes = ['Golden retriever', 'Rhodesian ridgeback', 'Australian terrier', 'Samoyed', 'Border terrier', 'Dingo',
               'Shih-Tzu', 'Beagle', 'English foxhound', 'Old English sheepdog']

    train_directory = './data/imagewoof2-160/train'
    test_directory = './data/imagewoof2-160/val'

    image_size_test = ImageFolder(train_directory, transforms.ToTensor())

    train_tfms = transforms.Compose([transforms.Resize([SIZE, SIZE]), transforms.ToTensor()])
    test_tfms = transforms.Compose([transforms.Resize([SIZE, SIZE]), transforms.ToTensor()])

    trainset = ImageFolder(data_dir + '/train', train_tfms)
    testset = ImageFolder(data_dir + '/val', test_tfms)

    classes_dict = dict(zip(os.listdir('./data/imagewoof2-160/train'), classes))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)




###################################
###      Helper Functions       ###
###################################

class BinarizerSTE(nn.Module):
    def __init__(self, input_channels=128, height=8, width=8):
        #  CIFAR 100 128, 8 ,8
        # Imagewoof 128, 16, 16
        super(BinarizerSTE, self).__init__()

        self.input_dim = input_channels * height * width

        self.channels = input_channels
        self.height = height
        self.width = width
        self.fc = nn.Linear(self.input_dim, self.input_dim)

    def forward(self, x):
        # First step: fully connected layer with tanh activation
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc(x))

        # Stochastic binarization
        rand = torch.rand_like(x)
        binarized_output = torch.where(rand < (1 + x) / 2, 1.0, -1.0)

        # Use straight through gradient estimator
        binarized_output = (binarized_output - x).detach() + x

        # Reshape to original dimensions
        binarized_output = binarized_output.view(x.size(0), self.channels, self.height, self.width)

        return binarized_output

    def forward_fixed(self, x):
        # Fixed representation for the forward pass after training
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc(x))
        return torch.where(x < 0, -1.0, 1.0)


# Straight-Through Estimator (STE) function for quantization


###############################################
# Convolutional Autoencoder with Quantization #
###############################################
" operates on the entire image"

class ConvAutoencoderQ(nn.Module):
    def __init__(self):
        super(ConvAutoencoderQ, self).__init__()

        # Encoder layers

        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv0 = nn.Conv2d(3, 16, 3, padding=1)
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv1 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv1k = nn.Conv2d(32, 64, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)

        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        # self.latent =

        self.quantize = BinarizerSTE()
        # Decoder layers
        self.t_conv_up0 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.t_conv_mix0 = nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1)
        self.t_conv_up1 = nn.ConvTranspose2d(32, 3, 2, stride=2)
        self.t_conv_mix1 = nn.ConvTranspose2d(3, 3, 3, stride=1, padding=1)



    def forward(self, x):
        # Encode
        # input shape is batch_size 3, 32 ,32
        x = F.relu(self.conv0(x))
        # x.shape is batch_size 16,32,32
        x = self.pool(x)
        # [batchsize, 16, 16 ,16]
        x = F.relu(self.conv1(x))
        # [batchsize, 32, 16, 16]
        x = self.pool(x)
        # [batchsize, 64, 8, 8]
        x = self.conv1k(x)
        x = F.relu(self.conv2(x))
        # torch.Size([32, 128, 8, 8])
        x = x-x.mean() #normalization

        # Latent space quantization
        x = self.quantize(x)

        # Decode
        x = F.relu(self.t_conv_up0(x))
        x = F.relu(self.t_conv_mix0(x))
        x = F.relu(self.t_conv_up1(x))
        x = torch.sigmoid(self.t_conv_mix1(x))

        return x


# Define the Convolutional Residual Autoencoder with Quantization
class ConvResidualAutoEncoderQ(nn.Module):
    def __init__(self, num_passes=7, pt_block=None):
        super(ConvResidualAutoEncoderQ, self).__init__()

        self.num_passes = num_passes
        self.blocks = nn.ModuleList([ConvAutoencoderQ() for p in range(num_passes)])


        if pt_block:
            self.blocks[0] = torch.load(pt_block)

    def forward(self, x):
        losses = []  # loss per each pass
        in_x = x.clone()
        residual_outputs = []

        sum_residual = torch.zeros_like(in_x)

        # Go through all residuals
        for i, b in enumerate(self.blocks):
            x = b(x)
            sum_residual += x
            residual_outputs.append((1/(i+1))*sum_residual.clone())

            if i < self.num_passes - 1:
                x = in_x - x

        return (1 / self.num_passes) * sum(residual_outputs), residual_outputs

    def tap_in(self, x):
        in_x = x.clone()
        residual_outputs = []

        # Go through all residuals
        for i, b in enumerate(self.blocks):
            x = b(x)
            residual_outputs.append(x.clone())

            if i == self.num_passes - 1:
                break

            x = in_x - x

        return (1 / self.num_passes) * (sum(residual_outputs)), residual_outputs

def get_accuracy( gt, preds):
    """
    Calculate the accuracy of predictions.

    Args:
        ground_truth (Tensor): Ground truth labels.
        predictions (Tensor): Predicted labels.

    Returns:
        int: Number of correct predictions.
    """
    pred_vals = torch.max(preds.data, 1)[1]
    batch_correct = (pred_vals == gt).sum().item()
    return batch_correct

def train_classifier(model,residual_autoencoder, train_loader, test_loader, criterion, optimizer, num_epochs=3, device='cuda'):
    torch.cuda.empty_cache()

    model.to(device)
    print(f'-------------------------------------')
    print(f' Began Training the Classifier')
    print(f'------------------------------------- \n' )

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        # Initialize the occurrence tensor with the correct shape
        occurrence = torch.zeros([residual_autoencoder.num_passes], dtype=torch.int32)

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)



            with torch.no_grad():
                recon_image, res_outputs = residual_autoencoder(inputs)
                indices = torch.randint(low=0, high=len(res_outputs), size=(BATCH_SIZE, 1))

                # Increment occurrence counts
                occurrence += torch.bincount(indices.view(-1), minlength=residual_autoencoder.num_passes)

                # Create noisy inputs
                noisy_inputs = []
                for ii in range(BATCH_SIZE):
                    noisy_inputs.append(res_outputs[indices[ii]][ii, :, :, :])

                noisy_inputs = torch.stack(noisy_inputs, dim=0)

            optimizer.zero_grad()

            outputs = model(noisy_inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += get_accuracy(labels, outputs)
            total_samples += labels.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples

        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            running_test_correct = 0.0
            for x_tests, y_labels in test_loader:
                x_tests, y_labels = x_tests.to(device), y_labels.to(device)
                x_recon, _ = residual_autoencoder(x_tests)
                y_prediction = model(x_recon)
                test_loss += criterion(y_prediction,y_labels)
                running_test_correct += get_accuracy(y_labels,y_prediction)


        print(f"Epoch {epoch + 1}/{num_epochs}, Traing Loss: {epoch_loss:.4f},"
              f"Test Loss: {test_loss/(BATCH_SIZE*len(test_loader)):.4f} \n, Training Accuracy: {100 * epoch_acc:.4f}, Test Accuracy: {100*running_test_correct/(BATCH_SIZE*len(test_loader))}:.4f")

    # Create a histogram using Matplotlib
    plt.figure(figsize=(10, 6))
    plt.bar(torch.arange(1, len(occurrence)+1).cpu().numpy(), occurrence.cpu().numpy(), color='blue', alpha=0.7)
    plt.xlabel('Index')
    plt.ylabel('Count')
    plt.title('Histogram of Index Occurrences')
    plt.grid(True)
    plt.show()


    print("Training complete")
    return model

def train(model, criterion, n_epochs=EPOCHS, lr=LEARNING_RATE, train_loader=trainloader, test_loader=testloader,
          overfit=False):
    """
    Train the model.

    Parameters:
    model (nn.Module): The model to train.
    criterion (nn.Module, optional): The loss function.
    n_epochs (int, optional): The number of epochs.
    lr (float, optional): The learning rate.
    train_loader (DataLoader, optional): The data loader.
    overfit (bool, optional): If True, overfit the model.
    """

    print(f'-------------------------------------')
    print(f' Began Training the Residual AutoEncoder')
    print(f'------------------------------------- \n')

    torch.cuda.empty_cache()
    scaler = torch.cuda.amp.GradScaler()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    accumulation_steps = 10
    for epoch in range(1, n_epochs + 1):
        torch.cuda.empty_cache()
        model.train()
        start_time = time.time()
        train_loss = 0.0

        optimizer.zero_grad()
        for batch_num, (Train, Labels) in enumerate(train_loader):
            batch_num += 1
            batch = Train.to(device)
            loss_per_pass = []

            with torch.cuda.amp.autocast():
                outputs, recon_per_pass = model(batch)
                loss = criterion(outputs, batch)  # Signal Reconstruction hence MSE loss between predicted image and true image

            scaler.scale(loss).backward()


            if batch_num % accumulation_steps == 0 or batch_num == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item()


            if batch_num % 100 == 0:

                print(
                    f'epoch: {epoch:2}  batch: {batch_num:2} [{BATCH_SIZE * batch_num:6}/{len(train_loader)}]  total loss: {loss.item():10.8f}  \
                           time = [{(time.time() - start_time) / 60}] minutes')


                if epoch == 5 or epoch == 9:
                    # Plot the reconstructed images for each pass
                    flag = True
                    count = 1
                    if flag:
                        fig, axes = plt.subplots(1, len(recon_per_pass) + 1, figsize=(15, 5))
                        for ii in range(len(recon_per_pass)):
                            axes[ii].imshow(recon_per_pass[ii][1].permute(1, 2, 0).detach().cpu().numpy())
                            axes[ii].set_title(f'Pass {ii + 1}')
                        # Plot the original image for comparison
                        axes[-1].imshow(batch[1].permute(1, 2, 0).detach().cpu().numpy())
                        axes[-1].set_title('Original')
                        plt.show()
                        count = count +1

                        if count == 3:
                            flag = False




        # Inference loop
        model.eval()
        with torch.no_grad():
            test_loss_val = 0.0
            for x_test, y_labels in test_loader:
                x_tests, y_labels = x_test.to(device), y_labels.to(device)
                x_reconstruction, _ = model(x_tests)
                test_loss_val += criterion(x_reconstruction, x_tests)

            test_loss_val = test_loss_val/(len(test_loader)*BATCH_SIZE)

        print(f'epoch: {epoch:2}   train loss: {train_loss/(BATCH_SIZE*len(train_loader)):10.8f} , test loss: {test_loss_val.item():10.8f},\
                                  time = [{(time.time() - start_time) / 60}] minutes')

    # Plot training loss
    # plt.figure()
    # plt.plot(range(1, n_epochs + 1), train_losses, label='Training Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Training Loss over Epochs')
    # plt.legend()
    # plt.show()

    return model, train_losses


inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 1], [6, 32, 3, 1], [6, 64, 4, 2], [6, 96, 3, 1],
                             [6, 160, 3, 1], [6, 320, 1, 1]]
#
mobilenetv2 = models.mobilenet_v2(pretrained=True, num_classes=1000, width_mult=1,
                                       inverted_residual_setting=inverted_residual_setting)

# Modify the classifier head to fit the number of classes in your dataset
mobilenetv2.classifier = nn.Sequential(
    nn.Dropout(0.1, inplace=True),
    nn.Linear(in_features=1280, out_features=NUM_CLASSES, bias=True)
)

# # Freeze all the parameters in the network
# for param in mobilenetv2.parameters():
#     param.requires_grad = False
#
# # Unfreeze the parameters of the classifier head
# for param in mobilenetv2.classifier.parameters():
#     param.requires_grad = True




# initialize the NN
residualQ_AE = ConvResidualAutoEncoderQ().to(device)
loss_func = nn.MSELoss()

# print(torch.cuda.memory_allocated(device='cuda:0'))
# print(torch.cuda.memory_reserved(device='cuda:0'))

train(residualQ_AE, criterion=loss_func, n_epochs=10)

# TODO: Need to change the classifier head to accommadate task-based settings
train_classifier(mobilenetv2,residualQ_AE, trainloader, testloader,  criterion=nn.CrossEntropyLoss(), optimizer=torch.optim.Adam(mobilenetv2.parameters(), lr=10*LEARNING_RATE),
                 num_epochs=20, device='cuda')



# Inference Loop

with torch.no_grad():
    accuracy_per_pass = [0.0] * residualQ_AE.num_passes

    for b, (X_test, y_test) in enumerate(testloader):
        # Apply the model
        b += 1
        batch = X_test.to(device)

        recon_images, recon_per_pass = residualQ_AE(batch)
        for j_pass in range(residualQ_AE.num_passes):
            prediction = mobilenetv2(recon_per_pass[j_pass])
            accuracy_per_pass[j_pass] += get_accuracy(y_test.to(device), prediction.to(device))


    test_acc = [100 * acc / (b*BATCH_SIZE) for acc in accuracy_per_pass]
    print(test_acc)
