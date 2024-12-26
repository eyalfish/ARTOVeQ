
###################################################################################################
###################################################################################################
#################################             Imports             #################################
###################################################################################################
###################################################################################################

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import ssl

from torch import optim
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

import utilis
import logging



# SEED = 13

# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



###################################################################################################
###################################################################################################
#################################   Globals & Hyperparameters     #################################
###################################################################################################
###################################################################################################

logging.basicConfig(level=logging.INFO)

BATCH_SIZE = 16 # Changed from 16


LEARNING_RATE = 1e-4
EPOCHS = 20
NUM_EMBED = 256 # Number of vectors in the codebook.
ARCH = 'IMAGEWOOF'
SIZE = 64 # 64 IS FOR IMAGEWOOF



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
    trainset = datasets.CIFAR100(root = path, train=True, download=True, transform=train_transform)
    testset = datasets.CIFAR100(root = path, train=False, download=True, transform=test_transform)


    # Define the size of the subset
        # train_subset_size = int(len(trainset) * 0.01)
        # test_subset_size = int(len(testset) * 0.01)
        #
        # # Create random indices for the subset
        # train_indices = np.random.choice(len(trainset), train_subset_size, replace=False)
        # test_indices = np.random.choice(len(testset), test_subset_size, replace=False)
        #
        # # Create the subset
        # train_subset = Subset(trainset, train_indices)
        # test_subset = Subset(testset, test_indices)

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


    with tarfile.open('./imagewoof2-160.tgz', 'r:gz') as tar: #read file in r mode
      tar.extractall(path = './data') #extract all folders from zip file and store under folder named data

    data_dir = './data/imagewoof2-160'
    # print(os.listdir(data_dir))
    # print(os.listdir('./data/imagewoof2-160/train'))
    # print(len(os.listdir('./data/imagewoof2-160/train')))
    classes = ['Golden retriever', 'Rhodesian ridgeback', 'Australian terrier', 'Samoyed', 'Border terrier', 'Dingo', 'Shih-Tzu', 'Beagle', 'English foxhound', 'Old English sheepdog']

    train_directory = './data/imagewoof2-160/train'
    test_directory = './data/imagewoof2-160/val'

    image_size_test = ImageFolder(train_directory, transforms.ToTensor())

    train_tfms = transforms.Compose([transforms.Resize([SIZE,SIZE]),transforms.ToTensor()])
    test_tfms = transforms.Compose([transforms.Resize([SIZE,SIZE]),transforms.ToTensor()])

    trainset = ImageFolder(data_dir + '/train', train_tfms)
    testset = ImageFolder(data_dir + '/val', test_tfms)

    classes_dict = dict(zip(os.listdir('./data/imagewoof2-160/train'), classes))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)
    total_samples = len(testset)

    num_samples = 2500
    subset_indices = torch.randperm(total_samples)[:num_samples]
    subset_dataset = Subset(testset, subset_indices)

    subset_testloader  = DataLoader(subset_dataset, batch_size=32, shuffle=True)


class AdaptiveVectorQuantizer(nn.Module):
    """
      Implements an adaptive vector quantization scheme. The codebook
      (representation vectors) are learned adaptively from low-resolution
      to high-resolution during training.
    """

    def __init__(self, num_embeddings: int, codebook_size: int, commitment_loss_weight=0.1, proximity_loss_weight=0.33):

        """
        Initialize the AdaptiveVectorQuantizer

        Args:
            num_embeddings (int): Size of the vectors.
            codebook_size (int): Number of codebook vectors.
            commitment_loss_weight (float, optional): Commitment loss parameter. Defaults to 0.1.
            proximity_loss_weight (float, optional): Balancing parameter of the proximity loss. Defaults to 0.33.
        """
        super(AdaptiveVectorQuantizer, self).__init__()

        self.d = num_embeddings  # The size of the vectors
        self.p = codebook_size  # Number of vectors in the codebook

        # initialize the codebook
        self.codebook = nn.Embedding(self.p, self.d)
        self.codebook.weight.data.uniform_(-1 / self.p, 1 / self.p) #initialze the codebook


        # Balancing parameter lambda for the commintment loss
        self.commitment_loss_weight = commitment_loss_weight
        self.proximity_loss_weight = proximity_loss_weight




    def forward(self, input_data, num_active_vectors, previous_active_vectors):
        """
        Foward pass of the Adaptive Vector Quantizer

        Args:
            inputs (Tensor): Input tensor of size B x C x H x W.
            num_active_vectors (Tensor): Number of active vectors for quantization.
            previous_active_vectors (Tensor): Previous active vectors.

        Returns:
            tuple: Tuple containing quantized vectors, losses, and active codebook vectors.
            Possible Edit: Include avg number of bits per image
        """

        #Input Preparation: Reshaping -> Flattening-> Quantization

        input_data = input_data.permute(0, 2, 3, 1).contiguous()  # Input is rearranged to BxHxWxC for easier processing
        input_shape= input_data.shape


        # Flatten input
        flat_input = input_data.view(-1, self.d)  # input vector is flattened to shape(-1,d)

        # Compute the variance of each segment

        quantized_vectors = []
        losses = []


        """The quantization process is performed for multiple levels of active vectors
        Normalization of the Euclidean distance ruins the performance of the model
        """


        for num_active_levels in range(int(np.log2(num_active_vectors))):
            active_vectors = self.codebook.weight[:pow(2, num_active_levels + 1)]

            # Calculate distance between each input vector and codebook vector


            distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                         + torch.sum(active_vectors ** 2, dim=1)
                         - 2 * torch.matmul(flat_input, active_vectors.t()))


            """
            Encoding:
            encoding indices are determined by finding the index of the nearest codebook vector
                j = argmin {||z_t - e_k||_2} for  k in {1,2,...,Q}
            """
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # finds the closest reference vector for each input vector.
            encodings = torch.zeros(encoding_indices.shape[0], self.p, device=input_data.device)
            encodings.scatter_(1, encoding_indices, 1)  # encoding indices are converted into one-hot encodings

            # Quantize and unflatten
            """
            The quantized vectors are obtained by a weighted sum of codebook vectors using
             the one-hot encodings.
            """
            quantized = torch.matmul(encodings, self.codebook.weight).view(input_shape)



            quantized_vectors.append(quantized)

        """ Loss Computation:
        q_latent_loss -> commitment loss: measure the difference between the quantized vectors to the encoded
            input vectors

        e_latent_loss -> alignment loss:

        prox_loss -> proximity loss: encourages nearby codebook vectors to be close to one another

        cb_loss -> overall codebook loss

        """
        for num_active_levels in range(int(np.log2(num_active_vectors))):

            if self.training:
                # Loss
                q_latent_loss = F.mse_loss(quantized_vectors[num_active_levels], input_data.detach())  # commitment loss


                if num_active_levels == 0:
                    prox_loss = 0
                    e_latent_loss = F.mse_loss(quantized_vectors[num_active_levels].detach(), input_data)  # alignment loss

                elif num_active_levels == 1:
                    e_latent_loss = F.mse_loss(quantized_vectors[num_active_levels].detach(), input_data)
                    prox_loss = (num_active_levels * self.proximity_loss_weight) * F.mse_loss(
                        previous_active_vectors[:pow(2, num_active_levels + 1) // 2],
                        active_vectors[:pow(2, num_active_levels + 1) // 2])

                else:
                    e_latent_loss = 0
                    prox_loss = self.proximity_loss_weight * F.mse_loss(previous_active_vectors[:pow(2, num_active_levels + 1) // 2],
                                                           active_vectors[
                                                           :pow(2, num_active_levels + 1) // 2])  # proximity_loss


                cb_loss = q_latent_loss + self.commitment_loss_weight * e_latent_loss + prox_loss  # codebook loss

                quantized_vectors[num_active_levels] = input_data + (
                            quantized_vectors[num_active_levels] - input_data).detach()  # gradient copying
                quantized_vectors[num_active_levels] = quantized_vectors[num_active_levels].permute(0, 3, 1,
                                                                                                    2).contiguous()

            else:
                # convert quantized from BHWC -> BCHW
                quantized_vectors[num_active_levels] = quantized_vectors[num_active_levels].permute(0, 3, 1,
                                                                                                    2).contiguous()

                # print(f'Distortion: {F.mse_loss(flat_input, quantized_vectors[num_active_levels].view(-1,8))}')
                cb_loss = 0

            losses.append(cb_loss)

        return quantized_vectors, losses, active_vectors

    def compute_encoding_per_section_per_quantization(self,flat_ze_section,quantization_level,ADC_resolution):

        if (quantization_level +1) > ADC_resolution:
            active_vectors_per_section = self.codebook.weight[:pow(2, ADC_resolution)]
        else:
            active_vectors_per_section = self.codebook.weight[:pow(2, quantization_level + 1)]

        distances_per_section_quantization_level = (torch.sum(flat_ze_section ** 2, dim=1, keepdim=True)
                                                         + torch.sum(active_vectors_per_section ** 2, dim=1)
                                                         - 2 * torch.matmul(flat_ze_section,
                                                                            active_vectors_per_section.t()))

        return torch.argmin(distances_per_section_quantization_level, dim=1).unsqueeze(1)


    def forward_with_split (self, input_data, num_active_vectors, previous_active_vectors, bit_resolutions):

        num_ADCs = len(bit_resolutions)

        #Input Preparation: Reshaping -> Flattening-> Quantization
        input_data = input_data.permute(0, 2, 3, 1).contiguous()  # Input is rearranged to BxHxWxC for easier processing
        input_shape = input_data.shape

        # Flatten input
        flat_input = input_data.view(-1, self.d)  # input vector is flattened to shape(-1,d)

        # Split the tensor into numADCs parts
        split_input = torch.chunk(flat_input,chunks=num_ADCs, dim=0) # returns list of tensors
        # active_vectors_per_section = [self.codebook.weight[:pow(2, bit_resolutions[kk])] for kk in range(num_ADCs)]


        quantized_vectors = []
        losses = []


        encoding_indices = []
        #Apply l2 norm on each section given bit resolution


        for num_active_levels in range(int(np.log2(num_active_vectors))):
            active_vectors = self.codebook.weight[:pow(2, num_active_levels + 1)]
            encoding_indices_section = []
            for ii in range(num_ADCs):
                # for each ADC bit resolution compute distances per quantization level

                encoding_indices_section.append(self.compute_encoding_per_section_per_quantization
                                                                 (split_input[ii],num_active_levels, bit_resolutions[ii]))


            encoding_indices = torch.cat(encoding_indices_section,dim=0)
            # encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # finds the closest reference vector for each input vector.


            encodings = torch.zeros(encoding_indices.shape[0], self.p, device=input_data.device)
            encodings.scatter_(1, encoding_indices, 1)  # encoding indices are converted into one-hot encodings

            # Quantize and unflatten
            """
            The quantized vectors are obtained by a weighted sum of codebook vectors using
             the one-hot encodings.
            """
            quantized = torch.matmul(encodings, self.codebook.weight).view(input_shape)
            quantized_vectors.append(quantized)

        # Loss computation

        for num_active_levels in range(int(np.log2(num_active_vectors))):

            if self.training:
                # Loss
                q_latent_loss = F.mse_loss(quantized_vectors[num_active_levels], input_data.detach())  # commitment loss


                if num_active_levels == 0:
                    prox_loss = 0
                    e_latent_loss = F.mse_loss(quantized_vectors[num_active_levels].detach(), input_data)  # alignment loss


                elif num_active_levels == 1:
                    e_latent_loss = F.mse_loss(quantized_vectors[num_active_levels].detach(), input_data)

                    prox_loss = (num_active_levels * self.proximity_loss_weight) * F.mse_loss(
                        previous_active_vectors[:pow(2, num_active_levels + 1) // 2],
                        active_vectors[:pow(2, num_active_levels + 1) // 2])

                else:
                    e_latent_loss = 0
                    prox_loss = self.proximity_loss_weight * F.mse_loss(previous_active_vectors[:pow(2, num_active_levels + 1) // 2],
                                                           active_vectors[
                                                           :pow(2, num_active_levels + 1) // 2])  # proximity_loss

                cb_loss = q_latent_loss + self.commitment_loss_weight * e_latent_loss + prox_loss  # codebook loss

                quantized_vectors[num_active_levels] = input_data + (
                            quantized_vectors[num_active_levels] - input_data).detach()  # gradient copying
                quantized_vectors[num_active_levels] = quantized_vectors[num_active_levels].permute(0, 3, 1,
                                                                                                    2).contiguous()

            else:
                # convert quantized from BHWC -> BCHW
                quantized_vectors[num_active_levels] = quantized_vectors[num_active_levels].permute(0, 3, 1,
                                                                                                    2).contiguous()
                cb_loss = 0


            losses.append(cb_loss)

        return quantized_vectors, losses,  active_vectors

    def forward_with_vqvae(self, input_data, active_vectors):
        """
                Foward pass of the Adaptive Vector Quantizer

                Args:
                    inputs (Tensor): Input tensor of size B x C x H x W.
                    num_active_vectors (Tensor): Number of active vectors for quantization.
                    previous_active_vectors (Tensor): Previous active vectors.

                Returns:
                    tuple: Tuple containing quantized vectors, losses, and active codebook vectors.
                    Possible Edit: Include avg number of bits per image
                """

        # Input Preparation: Reshaping -> Flattening-> Quantization

        input_data = input_data.permute(0, 2, 3, 1).contiguous()  # Input is rearranged to BxHxWxC for easier processing
        input_shape = input_data.shape

        # Flatten input
        flat_input = input_data.view(-1, self.d)  # input vector is flattened to shape(-1,d)

        # Compute the variance of each segment
        num_active_levels = int(math.log2(len(active_vectors)))

        quantized_vectors = []
        losses = []

        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(active_vectors ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, active_vectors.t()))

        """
        Encoding:
        encoding indices are determined by finding the index of the nearest codebook vector
            j = argmin {||z_t - e_k||_2} for  k in {1,2,...,Q}
        """
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(
            1)  # finds the closest reference vector for each input vector.
        encodings = torch.zeros(encoding_indices.shape[0], self.p, device=input_data.device)
        encodings.scatter_(1, encoding_indices, 1)  # encoding indices are converted into one-hot encodings

        # Quantize and unflatten
        """
        The quantized vectors are obtained by a weighted sum of codebook vectors using
         the one-hot encodings.
        """
        quantized = torch.matmul(encodings, self.codebook.weight).view(input_shape)
        quantized_vectors.append(quantized)

        # Loss computation
        if self.training:
            # Loss
            q_latent_loss = F.mse_loss(quantized_vectors[0], input_data.detach())  # commitment loss


            e_latent_loss = F.mse_loss(quantized_vectors[0].detach(), input_data)  # alignment loss


            cb_loss = q_latent_loss + self.commitment_loss_weight * e_latent_loss  # codebook loss

            quantized_vectors[0] = input_data + (
                    quantized_vectors[0] - input_data).detach()  # gradient copying
            quantized_vectors[0] = quantized_vectors[0].permute(0, 3, 1, 2).contiguous()

        else:
            # convert quantized from BHWC -> BCHW
            quantized_vectors[0] = quantized_vectors[0].permute(0, 3, 1, 2).contiguous()
            cb_loss = 0

        losses.append(cb_loss)

        return quantized_vectors, losses, active_vectors


##################################################################################################
##################################################################################################
#######################################       Model        #######################################
##################################################################################################
##################################################################################################

class AdapCB_Model(nn.Module):
    """
    Implements a model with adaptive vector quantization.
    """

    def __init__(self, num_embeddings, codebook_size, commitment_loss_weight=0.05, lambda_p=0.4, quant=True):
        super(AdapCB_Model, self).__init__()

        self.encoder, self.decoder, self.classifier = self.split_network()
        self.quantizer = AdaptiveVectorQuantizer(num_embeddings, codebook_size, commitment_loss_weight, lambda_p)
        self.quant = quant



    def build_model(self, pretrained=True, fine_tune=True):

        """
                Build the base model.

                Args:
                    pretrained (bool, optional): Use pre-trained weights. Defaults to True.
                    fine_tune (bool, optional): Fine-tune all layers. Defaults to True.
                Returns:
                    nn.Module: Built model.
        """

        if pretrained:
            print('[INFO]: Loading pre-trained weights')
        elif not pretrained:
            print('[INFO]: Not loading pre-trained weights')
        inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 1], [6, 32, 3, 1], [6, 64, 4, 2], [6, 96, 3, 1],
                                     [6, 160, 3, 1], [6, 320, 1, 1]]
        model = models.mobilenet_v2(pretrained=pretrained, num_classes=1000, width_mult=1,
                                    inverted_residual_setting=inverted_residual_setting)
        if fine_tune:
            print('[INFO]: Fine-tuning all layers...')
            for params in model.parameters():
                params.requires_grad = True
        elif not fine_tune:
            print('[INFO]: Freezing hidden layers...')
            for params in model.parameters():
                params.requires_grad = False

        # change the final classification head, it is trainable,
        model.dropout = nn.Dropout(0.1, inplace=True)
        model.fc = nn.Linear(in_features=FEATURES, out_features=NUM_CLASSES, bias=True)
        return model

    def split_network(self):
        """
        Splitting the Network
            Partitions the MobileNetV2 model into three parts: encoder, decoder, and quantizer

        """
        mobilenetv2 = self.build_model()

        encoder = []
        decoder = []
        classifier = []

        res_stop = 5
        for layer_idx, l in enumerate(mobilenetv2.features):
            if layer_idx <= res_stop:
                encoder.append(l)
            else:
                decoder.append(l)

        classifier.append(mobilenetv2.dropout)
        classifier.append(mobilenetv2.fc)

        Encoder = nn.Sequential(*encoder)
        Decoder = nn.Sequential(*decoder)
        Classifier = nn.Sequential(*classifier)
        return Encoder, Decoder, Classifier

    def get_accuracy(self, gt, preds):
        """
                Calculate the accuracy of predictions.

                Args:
                    ground_truth (Tensor): Ground truth labels.
                    predictions (Tensor): Predicted labels.

                Returns:
                    int: Number of correct predictions.
        """
        pred_vals = torch.max(preds.data, 1)[1]
        batch_correct = (pred_vals == gt).sum()
        return batch_correct

    def normalize(self, inputs):
        """
                Normalize the input.

                Args:
                    inputs (Tensor): Input tensor.

                Returns:
                    Tensor: Normalized input.
        """
        # Calculate the vector's magnitude
        mean = inputs.mean()

        output = inputs - mean
        return output

    def forward(self, inputs, num_active, previous_active_vectors):
        """
                Forward pass of the AdapCB_Model.

                Args:
                    inputs (Tensor): Input tensor.
                    num_active_vectors (Tensor): Number of active vectors for quantization.
                    prev_vectors (Tensor): Previous active vectors.

                Returns:
                    tuple: Tuple containing predictions, quantization loss, active codebook vectors, and original encoder output.
        """
        z_e = self.encoder(inputs)
        z_e = self.normalize(z_e)
        if self.quant == True:
            z_q, vq_loss, actives = self.quantizer.forward(z_e, num_active, previous_active_vectors)
        else:
            z_q, vq_loss, actives = [z_e], [0], None

            # Compute distortion between z_e and z_q
        #
        # print(f' Distortion between z_e and z_q using l2 norm {F.mse_loss(z_e.view(-1, self.quantizer.d).detach() , z_q[0].view(-1, self.quantizer.d).detach())}')

        preds_list = []
        for vecs in range(len(z_q)):
            z_q_actives = z_q[vecs]
            preds_list.append(self.decoder(z_q_actives))
            preds_list[vecs] = preds_list[vecs].reshape(preds_list[vecs].shape[0],
                                                        preds_list[vecs].shape[1] * preds_list[vecs].shape[2] *
                                                        preds_list[vecs].shape[3])
            preds_list[vecs] = self.classifier(preds_list[vecs])

        return preds_list, vq_loss, actives, z_e

    def forward_with_split(self, inputs,num_active_vectors, previous_active_vectors, bit_resolutions):

        z_e = self.encoder(inputs)
        z_e = self.normalize(z_e)
        # encoded signal is partitioned into 4 parts and mixed resolution codebooks are applied
        if self.quant == True:
            z_q, vq_loss, actives = self.quantizer.forward_with_split(z_e, num_active, previous_active_vectors, bit_resolutions)
        else:
            z_q, vq_loss, actives = [z_e], [0], None

        preds_list = []
        for vecs in range(len(z_q)):
            z_q_actives = z_q[vecs]
            preds_list.append(self.decoder(z_q_actives))
            preds_list[vecs] = preds_list[vecs].reshape(preds_list[vecs].shape[0],
                                                        preds_list[vecs].shape[1] * preds_list[vecs].shape[2] *
                                                        preds_list[vecs].shape[3])
            preds_list[vecs] = self.classifier(preds_list[vecs])

        return preds_list, vq_loss, actives, z_e


    def forward_with_vqvae(self,inputs, active_vectors):

        z_e = self.encoder(inputs)
        z_e = self.normalize(z_e)
        # encoded signal is partitioned into 4 parts and mixed resolution codebooks are applied
        if self.quant == True:
            z_q, vq_loss, actives = self.quantizer.forward_with_vqvae(z_e, active_vectors)
        else:
            z_q, vq_loss, actives = [z_e], [0], None

        preds_list = []
        for vecs in range(len(z_q)):
            z_q_actives = z_q[vecs]
            preds_list.append(self.decoder(z_q_actives))
            preds_list[vecs] = preds_list[vecs].reshape(preds_list[vecs].shape[0],
                                                        preds_list[vecs].shape[1] * preds_list[vecs].shape[2] *
                                                        preds_list[vecs].shape[3])
            preds_list[vecs] = self.classifier(preds_list[vecs])

        return preds_list, vq_loss, actives, z_e



###################################################################################################
###################################################################################################
##################################       Training Function        #################################
###################################################################################################
###################################################################################################


def train(model, optimizer, num_active, criterion, previous_active_vectors=None, EPOCHS=20, commitment=0.1,
          use_split=False, ADCs_resolutions=None, use_VQVAE=False):
    start_time = time.time()
    train_losses = []
    encoder_samples = []
    test_losses = []
    val_losses = 0
    stop_criteria = 55

    subset_indices = torch.randperm(len(trainset))
    subset = Subset(trainset, subset_indices)
    subset_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)

    if num_active == 2:
        loader = trainloader
    else:
        loader = subset_loader

    for epc in range(EPOCHS):

        if use_VQVAE == True:
            train_acc = [0.0]
            val_acc= [0.0]
        else:
            train_acc = [0] * int(np.log2(num_active))
            val_acc = [0] * int(np.log2(num_active))
            model.train()
            trn_corr = [0] * int(np.log2(num_active))

        model.train()
        losses = 0


        for batch_num, (Train, Labels) in enumerate(loader):
            batch_num += 1
            loss_levels = []
            batch = Train.to(device)

            if use_split == True and use_VQVAE == False and type(ADCs_resolutions) !=None:
                preds_list, vq_loss, curr_vecs, z_e = model.forward_with_split(batch, num_active, previous_active_vectors, ADCs_resolutions)
            if use_VQVAE == False and use_split== False:
                preds_list, vq_loss, curr_vecs, z_e = model.forward(batch, num_active, previous_active_vectors)

            if use_VQVAE == False:

                for q_level in range(len(preds_list)):
                    ce_loss = criterion(preds_list[q_level], Labels.to(device))
                    level_loss = ce_loss + vq_loss[q_level] # erased commitment parameter commitement* vq_loss[q_level]
                    loss_levels.append(level_loss)
                    train_acc[q_level] += model.get_accuracy(Labels.to(device), preds_list[q_level])
            else:
                preds_list, vq_loss, curr_vecs, z_e = model.forward_with_vqvae(batch, model.quantizer.codebook.weight)

                ce_loss = criterion(preds_list[0], Labels.to(device))
                level_loss = ce_loss + vq_loss[0]
                loss_levels.append(level_loss)
                train_acc[0] += model.get_accuracy(Labels.to(device), preds_list[0])



            loss = sum(loss_levels) / len(loss_levels)
            losses += loss.item()

            # Update parameters
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            if batch_num % 100 == 0:
                print(
                    f'epoch: {epc + 1:2}  batch: {batch_num:2} [{BATCH_SIZE * batch_num:6}/{len(trainset)}]  total loss: {loss.item():10.8f}  \
                               time = [{(time.time() - start_time) / 60}] minutes')

        ### Accuracy ###

        loss = losses / batch_num
        train_losses.append(loss)

        model.eval()
        test_losses_val = 0

        #inference loop
        with torch.no_grad():

            for b, (X_test, y_test) in enumerate(testloader):
                # Apply the model
                b += 1
                batch = X_test.to(device)


                if use_split == True:
                    val_preds, vq_val_loss, _, _ = model.forward_with_split(batch, num_active, previous_active_vectors, ADCs_resolutions)
                if use_VQVAE == False:
                    val_preds, vq_val_loss, _, _ = model.forward(batch, num_active, previous_active_vectors)
                loss_levels_val = []

                if use_VQVAE == False:

                    for q_level in range(len(val_preds)):
                        ce_val_loss = criterion(val_preds[q_level], y_test.to(device))
                        level_loss = ce_val_loss +  vq_val_loss[q_level] # Erased commitment * vq_loss
                        loss_levels_val.append(level_loss)
                        val_acc[q_level] += model.get_accuracy(y_test.to(device), val_preds[q_level])
                else:
                    preds_list, vq_loss, curr_vecs, z_e = model.forward_with_vqvae(batch, model.quantizer.codebook.weight)

                    ce_loss = criterion(preds_list[0], y_test.to(device))
                    level_loss = ce_loss +  vq_loss[0] # erased commitment * vq_loss
                    loss_levels_val.append(level_loss)
                    val_acc[0] += model.get_accuracy(y_test.to(device), preds_list[0])



                val_loss = sum(loss_levels_val)
                val_losses += val_loss.item()


            test_losses.append(val_losses / b)


        total_train_acc = [100 * (acc.item()) / len(trainset) for acc in train_acc]
        total_val_acc = [100 * (acc.item()) / len(testset) for acc in val_acc]
        # total_train_acc = 100*train_acc/len(trainset)
        # total_val_acc = 100*val_acc/len(testset)
        print(f'Train Models Accuracy at epoch {epc + 1} is {total_train_acc}%')
        print(f'Validation Models Accuracy at epoch {epc + 1} is {total_val_acc}%')
        model.train()

    encoder_samples = z_e
    encoder_samples = encoder_samples.permute(0, 2, 3, 1).contiguous()
    encoder_samples = encoder_samples.view(-1, 2)
    duration = time.time() - start_time
    print(f'Training took: {duration / 3600} hours')
    stop_criteria += 1
    return curr_vecs, encoder_samples


##################################################################################################
##################################################################################################
#################################           Helpers              #################################
##################################################################################################
##################################################################################################

def scatter(array, train_points):
    plt.rcParams["figure.figsize"] = (10, 10)
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'magenta', 'cyan', 'yellow']
    for i, level in enumerate(array):
        x = np.array([elem[0] for elem in level.detach().cpu()])
        y = np.array([elem[1] for elem in level.detach().cpu()])
        name = str(2 * pow(2, i)) + 'Bits Vectors'
        train_points_level = train_points[-1]
        train_points_level = train_points_level[:20000]
        train_x_vals = np.array([elem[0] for elem in train_points_level.detach().cpu()])
        train_y_vals = np.array([elem[1] for elem in train_points_level.detach().cpu()])
        plt.scatter(train_x_vals, train_y_vals, s=10, alpha=0.1, label='Train Vectors')
        plt.scatter(x, y, s=250, alpha=1, label=name, c=colors[i % 8])

        # Add axis labels and a title
        plt.xlabel('X',fontsize=12)
        plt.ylabel('Y',fontsize=12)
        plt.title('2D Scatter Plot',fontsize=12)
        plt.grid()
        plt.legend(loc='best', fontsize=12)
        # Show the plot
        plt.show()


def init_weights(m):
    if type(m) == nn.Embedding:
        torch.nn.init.normal_(m.weight, mean=0, std=0.5)



#################################################################################################
#################################################################################################
##############################      Train Adaptive Codebook       ###############################
#################################################################################################
#################################################################################################

cb_vec_dim = 4
previous_active_vectors = None  # Start with empty codebook
criterion = nn.CrossEntropyLoss()

mobilenetv2_ac = AdapCB_Model(num_embeddings=cb_vec_dim,
                              codebook_size=NUM_EMBED,
                              quant=False, lambda_p=0.33)


mobilenetv2_ac.to(device)


optimizer = torch.optim.Adam(mobilenetv2_ac.parameters(), lr=LEARNING_RATE) #weight_decay=1e-3

samples_for_scatter = []
vecs_to_save = []
EPOCHS = [10, 2, 2, 2, 2, 2, 2, 2]

for level in range(1):
    print(f' \n ------------------------------------- ')
    print(f' \n Number of embeddings {NUM_EMBED}: \t Quantization level {level + 1} ')
    print(f' \n ------------------------------------- ')

    num_active = pow(2, level + 1)  # Number of vectors-to-train in CB
    curr_vecs, encoder_samples = train(mobilenetv2_ac, optimizer, num_active, criterion, previous_active_vectors,
                                       EPOCHS[level])
    samples_for_scatter.append(encoder_samples)
    vecs_to_save.append(curr_vecs)




""" t-SNE """
embedding_samples = []
labels = []

with torch.no_grad():
    for images, label in subset_testloader:
        images = images.to(device)
        outputs = mobilenetv2_ac.encoder(images)  # Extract embeddings from the encoder

        # Flatten the outputs to 2D (batch_size, feature_dimension)
        flattened_outputs = outputs.view(outputs.size(0), -1)  # Flatten each sample

        embedding_samples.append(flattened_outputs.cpu().numpy())  # Convert to numpy and append
        labels.extend(label.numpy())

embedding_samples = np.vstack(embedding_samples)  # Convert list of arrays to a single numpy array
labels = np.array(labels)

# apply t-SNE

from sklearn.manifold import TSNE
print("Applying t-SNE...")
perplexities = [2, 3, 4, 5, 6]
for perplex_value in perplexities:
    print(f"Perplexity Value is {perplex_value}")
    tsne = TSNE(n_components=2, perplexity=perplex_value, random_state=42)
    tsne_results = tsne.fit_transform(embedding_samples)

    # Visualize the t-SNE results
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10', s=5)
    plt.colorbar(scatter, label='Class')
    plt.title(f't-SNE Visualization of MobileNetV2 Encodings: Perplexity {perplex_value}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()

pass
# print(f'\n t-SNE in 3d....')
# for perplex_value in perplexities:
#     print(f"Perplexity Value is {perplex_value}")
#     tsne = TSNE(n_components=3, perplexity=perplex_value, random_state=42)
#     tsne_results = tsne.fit_transform(embedding_samples)
#
#     # Create a 3D plot
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111, projection='3d')  # Add 3D subplot
#
#     # Scatter plot in 3D
#     scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], c=labels, cmap='tab10', s=5)
#
#     # Color bar
#     plt.colorbar(scatter, label='Class')
#
#     # Set titles and labels
#     ax.set_title('t-SNE Visualization of MobileNetV2 Encodings (3D)')
#     ax.set_xlabel('t-SNE Dimension 1')
#     ax.set_ylabel('t-SNE Dimension 2')
#     ax.set_zlabel('t-SNE Dimension 3')
#
#     plt.show()
#
# print(' finished tSNE')

pass

############################################################
############# Incorporating Kmeans quantization ############
############################################################

class Kmeans_Quantizer(nn.Module):

    def __init__(self, codebook):
        super(Kmeans_Quantizer, self).__init__()
        self.p, self.d = codebook.shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.active_vectors = codebook.to(self.device)

    def forward(self, z_e):
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        input_shape = z_e.shape
        flatten_ze = z_e.view(-1, self.d).to(self.device)

        distances = (torch.sum(flatten_ze ** 2, dim=1, keepdim=True)
                     + torch.sum(self.active_vectors ** 2, dim=1)
                     - 2 * torch.matmul(flatten_ze, self.active_vectors.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.active_vectors.shape[0], device=self.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.active_vectors).view(input_shape)
        return quantized.permute(0, 3, 1, 2).contiguous()

class Kmeans_Model(nn.Module):
    def __init__(self,encoder, decoder, classifer, quantizer):
        super(Kmeans_Model, self).__init__()
        self.encoder, self.decoder, self.classifier = encoder, decoder, classifer
        self.quantizer = quantizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def forward(self, x):

        z_e = self.encoder(x)
        z_e = z_e - z_e.mean()
        z_q = self.quantizer.forward(z_e)  #TODO: Needs to be a list datatype
        z_q = [z_q]
        preds_list = []

        for vecs in range(len(z_q)):
            z_q_actives = z_q[vecs]
            preds_list.append(self.decoder(z_q_actives))
            preds_list[vecs] = preds_list[vecs].reshape(preds_list[vecs].shape[0],
                                                        preds_list[vecs].shape[1] * preds_list[vecs].shape[2] *
                                                        preds_list[vecs].shape[3])
            preds_list[vecs] = self.classifier(preds_list[vecs])


        return preds_list

    def get_accuracy(self, ground_truth, predictions):
        predict_values = torch.max(predictions.data, 1)[1]
        return (ground_truth == predict_values).sum().item() / ground_truth.size(0)

def get_kmeans_codebooks(encoder, input_data, cb_vec_dim, num_clusters, kmeans_kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        z_e = encoder(input_data.to(device))
    z_e = z_e - z_e.mean()
    z_e = z_e.permute(0, 2, 3, 1).contiguous()
    flatten_ze = z_e.view(-1, cb_vec_dim)

    kmeans = KMeans(num_clusters, **kmeans_kwargs)
    kmeans.fit(flatten_ze.cpu().numpy())
    return torch.Tensor(kmeans.cluster_centers_)

def get_n_batches(data_loader, num_batches):
    collected_batches = 0
    all_samples = []
    all_labels = []
    for samples, labels in data_loader:
        all_samples.append(samples)
        all_labels.append(labels)
        collected_batches += 1
        if collected_batches == num_batches:
            break
    all_samples_tensor = torch.cat(all_samples, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    return all_samples_tensor, all_labels_tensor


# K-Means Quantization Setup
kmeans_kwargs = {
    "init": "random",
    "n_init": 2,
    "max_iter": 300,
}


NUM_CLUSTER = 2
k_means_quantizer = Kmeans_Quantizer(torch.zeros(NUM_EMBED, cb_vec_dim))  # Initialize with dummy codebook
model = Kmeans_Model(mobilenetv2_ac.encoder, mobilenetv2_ac.decoder, mobilenetv2_ac.classifier,k_means_quantizer.to(device))


# Generating Codebooks for Different Quantization Levels
input_data, _ = get_n_batches(trainloader, num_batches=10)
NUM_CLUSTERS = [2, 4, 8, 16, 32, 64, 128, 256]
CODEBOOKS = {num_clusters:
                 get_kmeans_codebooks(model.encoder, input_data, cb_vec_dim, num_clusters, kmeans_kwargs) for num_clusters in NUM_CLUSTERS }


# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.classifier.parameters()), lr=1e-4)

# Train only the decoder
for param in model.encoder.parameters():
    param.requires_grad = False

optimizer = optim.Adam(list(model.decoder.parameters()) + list(model.classifier.parameters()), lr=1e-4)


# Training Loop
num_epochs = 6
recluster_interval = 2  # Recompute KMeans every 2 epochs

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch + 1}/{num_epochs}')

    # Train the classifier with the current quantization level
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):

        selected_codebook = np.random.choice([2, 4, 8, 16, 32, 64, 128, 256])
        # if i < 10:
        #     print(selected_codebook)

        model.quantizer.active_vectors = torch.Tensor(CODEBOOKS[selected_codebook]).to(device)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs[0], labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print(f'Step [{i + 1}/{len(trainloader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

    # Evaluate on the test set for this quantization level
model.eval()





accuracy_per_q_level = []
for q_level in NUM_CLUSTERS:
    print(f' Quantization level: {q_level}')
    model.quantizer.active_vectors = torch.Tensor(CODEBOOKS[q_level]).to(device)
    test_accuracy = 0.0
    with torch.no_grad():
        for index, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            preds_list = model.forward(inputs)
            test_accuracy += model.get_accuracy(labels.to(device), preds_list[0])
    accuracy_per_q_level.append(100.0*test_accuracy/index)

print(f'Test accuracy  {accuracy_per_q_level}')

