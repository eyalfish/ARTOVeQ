###################################################################################################
###################################################################################################
#################################        Adaptive Codebook        #################################
################################################################################################
###################################################################################################

''' Here the implementation of Learning Multi-Rate Vector Quantization for Remote Deep Inference by
May Malka, Shai Ginzach, and Nir Shlezinger

For further questions: maymal@post.bgu.ac.il
'''

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
NUM_EMBED = 8 # Number of vectors in the codebook.
ARCH =  'CIFAR'
SIZE = 64 # 64 IS FOR IMAGEWOOF
# OHAD_WAS_HERE = True
# Eyal_is_here = False


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



###################################################################################################
###################################################################################################
#################################       Adaptive Quantizer        #################################
###################################################################################################
###################################################################################################

# images, labels = next(iter(testset))
#
# # Plot the first 16 images in a 4x4 grid
# fig, axes = plt.subplots(4, 4, figsize=(10, 10))
#
# for i, ax in enumerate(axes.flat):
#     img = images[i].permute(1, 2, 0).numpy()  # Convert to numpy array and rearrange dimensions
#     ax.imshow(img)
#     ax.axis('off')
#
# plt.show()

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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




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
        flat_input = input_data.view(-1, self.d) # input vector is flattened to shape(-1,d)
        # flat_input = flat_input + torch.normal(mean=0, std= 1, size = flat_input.shape).to(self.device)  # sanity check added noise

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


def init_weights_lbg(module, cb_vec_dim,codebook_length):
    if isinstance(module, nn.Embedding):

        if cb_vec_dim == 2:
            if codebook_length == 2:
                weight_tensor = torch.Tensor([[0.5130, 0.1811],
                        [-0.5280, -0.1852]])
                module.weight = nn.Parameter(weight_tensor)
                return module.weight

            if codebook_length == 4:
                weight_tensor =  torch.Tensor([[0.4104, -0.5343],
                                     [0.6973, 0.5062],
                                     [-0.7388, -0.4614],
                                     [-0.3550, 0.5219]])
                module.weight = nn.Parameter(weight_tensor)
                return module.weight

            if codebook_length == 8:
             weight_tensor = torch.Tensor([[0.3388, 0.0497],
                                 [0.8514, 0.7508],
                                 [-0.3362, -0.0784],
                                 [0.1278, -0.8689],
                                 [-0.9899, -0.7312],
                                 [1.1257, -0.4249],
                                 [-1.0402, 0.3866],
                                 [-0.1310, 0.8814]])
             module.weight = nn.Parameter(weight_tensor)

             return module.weight

            if codebook_length == 16:
             weight_tensor =  torch.Tensor([[-0.6976, 1.1572],
                         [-0.1028, -0.5541],
                         [-1.0868, 0.3357],
                         [-0.7611, -0.4203],
                         [-1.5911, -0.5542],
                         [0.7927, 0.0918],
                         [1.4753, -0.4999],
                         [1.1817, 0.7603],
                         [0.3802, 0.5871],
                         [-0.6152, -1.2090],
                         [0.5502, -0.4818],
                         [-0.2443, 0.5889],
                         [-0.4139, 0.0322],
                         [0.3054, 1.3390],
                         [0.1448, 0.0245],
                         [0.4331, -1.2072]])
             module.weight = nn.Parameter(weight_tensor)
             return module.weight

            if codebook_length == 32:
                 weight_tensor = torch.Tensor([[0.4905, 1.6717],
                                      [0.5937, -0.4158],
                                      [-1.2960, 0.8263],
                                      [0.0375, 1.0585],
                                      [1.6821, 0.2261],
                                      [-0.2112, -0.1175],
                                      [-0.0393, 0.5774],
                                      [-0.6223, -0.1321],
                                      [-1.9522, -0.1490],
                                      [0.1002, 0.1427],
                                      [-0.1368, -1.0867],
                                      [-0.6767, -0.8902],
                                      [1.7118, -0.7459],
                                      [0.1552, -0.3037],
                                      [-0.5530, 1.5341],
                                      [-0.9327, -0.4705],
                                      [-0.2813, -0.5393],
                                      [-0.5245, 0.8105],
                                      [-0.8204, 0.3465],
                                      [-0.7454, -1.6202],
                                      [0.2643, -0.7800],
                                      [0.5407, 0.0331],
                                      [-1.2325, -0.0081],
                                      [-0.3575, 0.2794],
                                      [0.9272, 0.3950],
                                      [0.5956, 0.9101],
                                      [0.4566, -1.5210],
                                      [0.3950, 0.4745],
                                      [1.0714, -0.2035],
                                      [0.8967, -0.8972],
                                      [-1.4887, -0.9614],
                                      [1.2537, 1.0472]])
                 module.weight = nn.Parameter(weight_tensor)
                 return  module.weight

            if codebook_length == 64:
                 weight_tensor = torch.Tensor( [[0.7013, 1.1442],
                                      [1.2806, 0.3924],
                                      [-0.8798, 1.7192],
                                      [-0.1079, -0.3376],
                                      [0.5855, -0.4643],
                                      [-0.7912, -1.6401],
                                      [-0.5276, 0.0175],
                                      [-0.0150, 0.2133],
                                      [1.1203, 0.8495],
                                      [-0.6816, -0.6081],
                                      [1.2820, 1.4654],
                                      [0.1002, -0.0874],
                                      [-1.5277, -0.7499],
                                      [-0.5055, -1.0129],
                                      [0.0281, -0.6134],
                                      [1.2031, -0.0684],
                                      [-0.1063, 1.8420],
                                      [-0.1661, -1.3755],
                                      [0.2266, 1.2602],
                                      [0.3542, -1.2146],
                                      [2.1122, -0.8431],
                                      [1.5025, -0.4808],
                                      [-1.3644, 1.0046],
                                      [-1.6034, -1.4639],
                                      [-1.2512, 0.3793],
                                      [0.2728, 0.1816],
                                      [0.2350, -0.3864],
                                      [0.3495, -0.7627],
                                      [0.5759, 0.1286],
                                      [0.6199, 1.8495],
                                      [-0.7479, 1.0846],
                                      [0.7896, -0.1412],
                                      [0.7477, -0.8941],
                                      [-0.0081, 0.8584],
                                      [-0.8650, 0.6371],
                                      [-0.2742, 1.2320],
                                      [-0.0420, -0.9488],
                                      [0.4866, 0.4450],
                                      [-0.8216, 0.2005],
                                      [0.4204, -0.1382],
                                      [-0.3041, 0.2262],
                                      [-0.7639, -0.2273],
                                      [-1.0851, -0.4914],
                                      [-0.3117, -0.6407],
                                      [-0.5376, 0.4219],
                                      [-1.5542, -0.1622],
                                      [-0.4162, 0.7867],
                                      [0.7693, 0.6542],
                                      [1.7689, 0.8133],
                                      [1.9152, 0.0567],
                                      [-0.1715, 0.5209],
                                      [-0.9954, -1.0024],
                                      [1.3197, -1.0273],
                                      [0.1844, -1.9463],
                                      [-0.2099, -0.0592],
                                      [0.9916, -0.5100],
                                      [0.8745, 0.2666],
                                      [0.1590, 0.5100],
                                      [-0.4320, -0.3121],
                                      [-1.9077, 0.4222],
                                      [0.3853, 0.8167],
                                      [-1.1028, -0.0577],
                                      [0.8698, -1.5247],
                                      [-2.2645, -0.5188]])

                 module.weight = nn.Parameter(weight_tensor)
                 return module.weight


            if codebook_length == 128:
                weight_tensor = torch.Tensor([[1.5392e-01, 1.3426e+00],
                          [-6.6599e-01, 1.5591e+00],
                          [5.3407e-01, -1.4238e+00],
                          [4.2723e-01, -6.9875e-02],
                          [-1.3912e-01, 6.0203e-01],
                          [5.7235e-02, -9.4828e-01],
                          [5.7202e-01, 1.1483e-01],
                          [9.8222e-01, 4.0903e-01],
                          [-2.6610e-01, 4.0540e-01],
                          [2.9971e-01, 7.1266e-01],
                          [-4.6051e-01, -8.5284e-01],
                          [-4.1606e-01, 6.0728e-01],
                          [4.7664e-01, -3.0928e-01],
                          [5.2026e-01, -8.3324e-01],
                          [-1.4497e+00, 5.4350e-01],
                          [1.2973e+00, 4.3214e-01],
                          [-7.0700e-01, 2.2183e+00],
                          [-7.7081e-02, 1.0697e+00],
                          [-1.0129e+00, 1.1380e-01],
                          [2.1632e+00, 7.1654e-01],
                          [4.4001e-02, 1.1549e-01],
                          [-7.9124e-01, 2.7655e-01],
                          [-7.6457e-01, -8.3302e-01],
                          [5.0344e-01, -5.6011e-01],
                          [-4.9448e-01, -2.1011e+00],
                          [1.6225e+00, -2.4511e-01],
                          [-1.3310e+00, 1.7949e-01],
                          [-2.0440e+00, -1.1889e+00],
                          [-3.1488e-01, -1.8021e-01],
                          [6.6046e-01, -1.2094e-01],
                          [4.9174e-01, 3.2846e-01],
                          [-1.2168e+00, -1.2691e-01],
                          [-1.6389e-01, 1.7979e-01],
                          [2.3930e-01, 9.9472e-01],
                          [-1.5591e+00, -7.5318e-01],
                          [2.5681e-01, -2.3627e-01],
                          [-1.3327e+00, 1.5150e+00],
                          [7.2121e-01, -1.0569e+00],
                          [4.9364e-01, 1.1868e+00],
                          [-3.7525e-01, 2.2128e-01],
                          [8.1035e-01, 1.1279e-01],
                          [1.2033e+00, -2.2609e-01],
                          [-4.8582e-01, -3.5506e-01],
                          [1.1091e+00, 7.5912e-01],
                          [1.3665e+00, -5.6206e-01],
                          [-3.1375e-01, -1.5026e+00],
                          [3.1939e-02, 7.9580e-01],
                          [-5.3309e-01, 3.7398e-01],
                          [7.6547e-02, -1.3097e+00],
                          [-1.9630e+00, -4.3006e-01],
                          [-1.7438e+00, 1.3779e-01],
                          [-5.4370e-01, -1.2034e-01],
                          [7.5519e-02, -3.8136e-01],
                          [-3.6467e-01, -5.9911e-01],
                          [1.0321e+00, -8.2867e-01],
                          [3.8993e-01, 2.3452e+00],
                          [7.2274e-01, -3.8251e-01],
                          [3.1864e-01, 4.5883e-01],
                          [1.0528e+00, 1.8815e+00],
                          [-5.8114e-01, -1.1722e+00],
                          [-8.9772e-01, -5.3700e-01],
                          [4.4124e-01, 1.6607e+00],
                          [-7.1435e-01, 5.4950e-01],
                          [7.5587e-01, -6.7873e-01],
                          [-2.2927e+00, 2.3142e-01],
                          [-1.3901e+00, -1.2571e+00],
                          [5.9586e-02, -1.3124e-01],
                          [-5.8648e-01, 8.3358e-01],
                          [3.3755e-01, 1.6954e-01],
                          [6.8954e-02, -6.4832e-01],
                          [1.3848e+00, 6.8041e-02],
                          [-5.8084e-01, 1.2159e-01],
                          [5.4875e-01, -2.5303e+00],
                          [2.1437e+00, -4.9816e-02],
                          [-2.6523e-01, 8.4217e-01],
                          [1.8119e+00, -1.3001e+00],
                          [-4.4279e-01, 1.1165e+00],
                          [5.3834e-01, 8.5861e-01],
                          [-3.5126e-01, 2.1732e-02],
                          [8.2630e-01, 9.9001e-01],
                          [2.5766e+00, -7.8119e-01],
                          [-1.1730e-01, -5.2456e-01],
                          [-2.3117e-01, -1.0953e+00],
                          [8.1273e-01, 1.3791e+00],
                          [-2.6184e+00, -5.3526e-01],
                          [1.2076e+00, 1.1988e+00],
                          [-7.1862e-01, -3.0707e-01],
                          [-9.1574e-01, 7.6669e-01],
                          [-1.8560e+00, 8.9328e-01],
                          [9.1336e-01, -1.6162e-01],
                          [-3.7768e-02, 3.5912e-01],
                          [7.2351e-01, 3.6406e-01],
                          [1.6880e-01, 2.9535e-01],
                          [-1.3223e-01, -3.6901e-02],
                          [-9.6056e-01, -1.1237e+00],
                          [-8.8227e-01, -1.5572e+00],
                          [-1.5349e+00, -2.6332e-01],
                          [1.7263e+00, 2.9926e-01],
                          [1.7958e-01, -1.7643e+00],
                          [-1.0630e-01, -2.6356e-01],
                          [-1.2163e+00, -4.6752e-01],
                          [9.7143e-01, -1.7991e+00],
                          [-2.3592e-01, 1.3927e+00],
                          [-1.2522e+00, 9.3180e-01],
                          [1.8504e+00, -6.7596e-01],
                          [3.4488e-01, -1.0820e+00],
                          [-2.7577e-01, -3.8718e-01],
                          [2.7626e-01, -4.8428e-01],
                          [2.1994e-01, -2.4867e-03],
                          [-8.3546e-01, 1.1413e+00],
                          [1.0740e+00, 1.1190e-01],
                          [1.0629e+00, -1.2725e+00],
                          [-9.6124e-01, -2.3465e-01],
                          [1.5353e+00, 8.0515e-01],
                          [-1.4609e+00, -1.9468e+00],
                          [-1.7517e-01, -7.8537e-01],
                          [8.0181e-01, 6.6162e-01],
                          [-1.0596e+00, 4.5788e-01],
                          [1.7515e+00, 1.4839e+00],
                          [-1.1354e+00, -8.0790e-01],
                          [-8.7848e-02, 1.8070e+00],
                          [5.4320e-01, 5.7776e-01],
                          [2.8007e-01, -7.5805e-01],
                          [1.0381e-01, 5.4466e-01],
                          [1.0011e+00, -4.7810e-01],
                          [-7.7923e-01, -2.6001e-02],
                          [1.4271e+00, -9.5415e-01],
                          [-6.1619e-01, -5.7627e-01]])
                module.weight = nn.Parameter(weight_tensor)
                return module.weight

            if codebook_length== 256:
                weight_tensor= torch.Tensor([[-1.0106e+00, -3.7316e-01],
                                     [-3.3072e-01, 1.1838e-01],
                                     [2.3951e-01, 5.0871e-02],
                                     [8.6485e-01, -6.9431e-01],
                                     [-7.0503e-01, 6.5650e-01],
                                     [1.0328e+00, 2.0625e+00],
                                     [-1.0105e+00, -1.0607e+00],
                                     [8.2660e-01, -3.0702e-02],
                                     [-3.4399e-01, -2.5088e-02],
                                     [4.8542e-01, -6.3078e-01],
                                     [9.6543e-01, -1.1749e+00],
                                     [-1.8873e-02, -5.6114e-01],
                                     [-6.1925e-01, 8.7481e-01],
                                     [-1.2176e+00, 4.8018e-02],
                                     [4.8373e-01, 1.6035e+00],
                                     [-1.5277e+00, -4.0218e-01],
                                     [8.7131e-01, -4.6806e-01],
                                     [9.8410e-01, 1.3959e+00],
                                     [7.9847e-01, 3.5055e-01],
                                     [-4.9577e-01, -7.2052e-01],
                                     [-1.7893e+00, 1.0348e+00],
                                     [1.0469e+00, -8.6505e-01],
                                     [-8.5875e-02, -2.8521e-01],
                                     [1.6929e+00, 3.8353e-02],
                                     [-2.7127e+00, -7.8866e-01],
                                     [-2.2162e+00, -1.5303e+00],
                                     [-1.1208e+00, -1.6343e-01],
                                     [9.2139e-01, 6.9117e-01],
                                     [-2.0436e+00, 1.5545e-01],
                                     [-1.4556e-02, -1.3834e+00],
                                     [9.7050e-01, 4.8199e-01],
                                     [-1.2325e+00, -3.9267e-01],
                                     [-7.8393e-01, -9.9569e-04],
                                     [8.5506e-03, -1.1003e+00],
                                     [1.6636e+00, 2.1874e+00],
                                     [5.1643e-01, 1.1712e-01],
                                     [2.3161e+00, -6.3727e-01],
                                     [-8.5450e-01, 8.4911e-01],
                                     [1.3748e-03, 6.0890e-01],
                                     [3.1532e-01, 3.1605e-01],
                                     [4.9088e-01, 7.7400e-01],
                                     [8.8575e-02, 7.1166e-02],
                                     [-9.7617e-01, 3.8625e-02],
                                     [-1.3380e+00, -1.1374e+00],
                                     [7.0222e-01, -1.1868e+00],
                                     [5.5521e-01, -6.0085e-02],
                                     [-2.1568e-01, 1.0357e+00],
                                     [-4.1164e-01, 8.7270e-01],
                                     [-2.9693e-01, 2.6629e+00],
                                     [1.5448e+00, -8.6812e-01],
                                     [-6.5219e-01, -1.5125e+00],
                                     [1.1780e+00, 1.1613e+00],
                                     [1.4271e+00, 2.4476e-01],
                                     [1.8863e+00, -9.0943e-01],
                                     [-9.2996e-01, 3.7636e-01],
                                     [-3.1169e-01, 7.0060e-01],
                                     [2.9037e-01, -6.5653e-01],
                                     [6.1694e-01, 4.1520e-01],
                                     [7.3655e-03, 2.0351e-01],
                                     [1.5691e+00, 1.1857e+00],
                                     [2.5244e-01, 1.1537e+00],
                                     [-5.0161e-01, -1.1894e+00],
                                     [4.8300e-01, 2.8491e-01],
                                     [1.0071e+00, 6.4471e-02],
                                     [5.8582e-01, 2.5880e+00],
                                     [-8.4356e-01, 1.7941e+00],
                                     [2.6702e-01, 4.7273e-01],
                                     [6.3833e-01, -7.7608e-01],
                                     [2.4010e+00, 2.7621e-03],
                                     [-9.3000e-02, -7.4412e-01],
                                     [-1.5721e+00, -8.3102e-01],
                                     [3.5262e-01, -4.9530e-01],
                                     [-4.8520e-01, 4.3337e-02],
                                     [-1.0340e+00, 1.0674e+00],
                                     [-8.6339e-01, 1.3764e+00],
                                     [-2.0426e-01, -5.9785e-01],
                                     [8.3242e-01, 8.9457e-01],
                                     [-5.0441e-01, 6.8501e-01],
                                     [-2.2853e+00, 6.6043e-01],
                                     [-1.2727e-01, -1.3116e-01],
                                     [3.5310e-01, 1.6644e-01],
                                     [-4.7628e-01, -4.2027e-01],
                                     [-4.1313e-01, -1.8672e+00],
                                     [-1.3976e+00, 5.6968e-01],
                                     [-5.2529e-01, -9.1336e-01],
                                     [-4.4242e-01, 3.7028e-01],
                                     [-6.4422e-01, 1.1997e-01],
                                     [-2.2157e-01, -1.1922e+00],
                                     [1.7277e-01, 9.5501e-01],
                                     [1.3722e-02, -1.7123e+00],
                                     [-7.2228e-01, 1.1033e+00],
                                     [1.9674e-01, -1.2292e+00],
                                     [-2.1824e+00, -3.1914e-01],
                                     [-1.8041e+00, -5.4635e-01],
                                     [-2.9753e-01, -4.3995e-01],
                                     [1.1861e+00, -1.3119e-01],
                                     [-3.0755e-01, 2.6785e-01],
                                     [-7.2379e-01, -1.9346e-01],
                                     [-2.2258e-01, 8.5517e-01],
                                     [8.3506e-01, 1.6974e+00],
                                     [-1.0605e+00, -1.4040e+00],
                                     [7.9483e-01, -2.7196e+00],
                                     [-2.2623e-01, 1.2451e+00],
                                     [5.3688e-01, -4.4957e-01],
                                     [1.8142e+00, -5.6538e-01],
                                     [8.8930e-01, -1.8861e+00],
                                     [3.7460e-01, 6.1739e-01],
                                     [-6.5850e-01, -3.8687e-01],
                                     [-4.7846e-01, 2.0625e-01],
                                     [2.3452e+00, 7.9619e-01],
                                     [-4.4842e-01, -1.2940e-01],
                                     [-1.3860e+00, 8.8246e-01],
                                     [-6.9900e-01, -7.3755e-01],
                                     [-1.3495e+00, -6.2799e-01],
                                     [1.0720e+00, 8.9627e-01],
                                     [1.3010e+00, -3.7736e-01],
                                     [-1.3722e+00, -1.7888e-01],
                                     [-8.5499e-02, 4.6238e-01],
                                     [-2.0955e-01, 1.4881e+00],
                                     [-3.4269e-01, -9.9433e-01],
                                     [1.1569e-02, 3.4595e-01],
                                     [5.6050e-02, -8.9805e-01],
                                     [4.3214e-01, -1.8343e+00],
                                     [-3.2897e-02, 9.1596e-01],
                                     [-8.2854e-01, -5.5154e-01],
                                     [-1.3801e-01, 3.0251e-01],
                                     [-5.2400e-01, 1.6156e+00],
                                     [1.5677e-01, -5.2575e-01],
                                     [-1.1728e+00, 2.3136e+00],
                                     [-6.2745e-01, 3.0201e-01],
                                     [-2.5788e-01, 4.1892e-01],
                                     [9.7276e-02, 4.8158e-01],
                                     [-1.7536e+00, -1.4596e-01],
                                     [-9.1677e-01, 6.0337e-01],
                                     [2.8345e-01, -9.4481e-02],
                                     [1.2089e+00, 1.2812e-01],
                                     [1.3761e+00, 8.7087e-01],
                                     [-1.3377e+00, 1.2278e+00],
                                     [-6.0228e-01, -5.5682e-01],
                                     [4.4839e-01, -1.2369e+00],
                                     [9.6122e-03, 1.1240e+00],
                                     [-3.8931e-01, -5.8257e-01],
                                     [1.2817e+00, -1.0809e+00],
                                     [-3.8258e-01, 5.3574e-01],
                                     [-1.5207e-01, -9.3615e-01],
                                     [-1.0656e+00, 2.2879e-01],
                                     [1.8304e-01, 1.9579e-01],
                                     [-2.0847e+00, -8.6313e-01],
                                     [2.9601e-01, -2.3797e-01],
                                     [1.0011e+00, 2.7235e-01],
                                     [8.0160e-02, 2.1141e+00],
                                     [8.2545e-01, 1.6349e-01],
                                     [-2.0024e-01, 2.8903e-03],
                                     [1.4086e+00, -6.5852e-02],
                                     [1.1250e+00, -1.4662e+00],
                                     [9.8293e-01, -1.3062e-01],
                                     [3.9244e-01, 9.6319e-01],
                                     [3.2777e+00, -7.6131e-01],
                                     [-1.8520e+00, 1.6404e+00],
                                     [-1.1153e+00, 7.6239e-01],
                                     [1.0593e+00, -3.3296e-01],
                                     [7.6348e-01, 5.4264e-01],
                                     [1.4109e-01, -8.5136e-02],
                                     [1.5039e+00, -1.8927e+00],
                                     [-1.6867e-01, 1.7854e+00],
                                     [4.3602e-01, 4.4745e-01],
                                     [1.4946e+00, -5.6968e-01],
                                     [-7.3632e-01, -9.7634e-01],
                                     [6.8969e-02, -2.2226e+00],
                                     [1.3455e+00, 1.5751e+00],
                                     [1.5834e-01, 3.3605e-01],
                                     [5.8724e-02, 1.3603e+00],
                                     [-1.7828e-03, -5.4920e-02],
                                     [1.7808e+00, 7.7666e-01],
                                     [-5.1242e-01, 1.3185e+00],
                                     [6.4979e-01, 2.3480e-01],
                                     [2.8282e-01, -1.0186e+00],
                                     [-2.7524e-01, -1.5704e-01],
                                     [-2.6947e+00, -9.5596e-03],
                                     [-1.0373e-01, 7.3342e-01],
                                     [8.3486e-01, 1.1288e+00],
                                     [6.8357e-01, -5.7232e-01],
                                     [-1.7436e-01, 1.6369e-01],
                                     [-1.5334e+00, -2.1686e+00],
                                     [1.0817e+00, -5.6437e-01],
                                     [6.2125e-01, 9.4742e-01],
                                     [2.2197e+00, -1.2559e+00],
                                     [1.1610e+00, 6.4908e-01],
                                     [7.0096e-01, -3.6451e-01],
                                     [4.3195e-01, -1.5702e-01],
                                     [1.6019e+00, -2.9059e-01],
                                     [4.8756e-01, 1.9698e+00],
                                     [-1.0677e+00, -5.9354e-01],
                                     [-7.5328e-01, 4.3834e-01],
                                     [-1.7485e+00, -1.1827e+00],
                                     [-1.1310e+00, 4.8206e-01],
                                     [-6.6978e-02, 6.6966e-02],
                                     [-3.8503e-01, -2.8311e-01],
                                     [-6.8197e-01, -2.4065e+00],
                                     [4.2391e-01, -8.1972e-01],
                                     [-8.2643e-01, -3.4855e-01],
                                     [1.5151e-01, -2.4611e-01],
                                     [-1.3222e+00, 2.9176e-01],
                                     [5.2065e-01, 1.1545e+00],
                                     [2.0412e+00, 2.4443e-01],
                                     [5.6161e-01, 5.8999e-01],
                                     [-9.7105e-01, -1.7950e+00],
                                     [9.1513e-02, 7.7068e-01],
                                     [8.5105e-01, -2.5814e-01],
                                     [6.7865e-01, -1.4905e+00],
                                     [7.9375e-01, -9.3643e-01],
                                     [-8.1983e-01, 2.0623e-01],
                                     [-1.4575e+00, -1.5580e+00],
                                     [5.6235e-01, -2.5706e-01],
                                     [-3.0303e-01, -7.7503e-01],
                                     [3.9999e-01, 8.0143e-03],
                                     [-6.0844e-01, -6.6828e-02],
                                     [-1.9332e-01, 5.7987e-01],
                                     [1.9084e-01, 6.2546e-01],
                                     [-9.1187e-01, -1.6499e-01],
                                     [1.4858e-01, 1.6591e+00],
                                     [3.4305e-01, 1.3697e+00],
                                     [-9.2166e-01, -7.8937e-01],
                                     [1.2701e+00, -7.1969e-01],
                                     [-5.5351e-01, -2.4880e-01],
                                     [-1.1910e+00, -8.4410e-01],
                                     [5.3413e-01, -9.9667e-01],
                                     [-5.6969e-01, 4.9558e-01],
                                     [4.1009e-01, -3.3647e-01],
                                     [-2.2597e-01, -2.9950e-01],
                                     [9.5616e-02, -6.9728e-01],
                                     [6.7556e-01, 5.5507e-02],
                                     [-1.4898e+00, 4.3885e-02],
                                     [6.6756e-01, 1.3613e+00],
                                     [-1.2383e-01, -4.3867e-01],
                                     [1.1969e+00, 3.9142e-01],
                                     [2.2875e-01, -8.2950e-01],
                                     [1.6961e+00, 4.0224e-01],
                                     [-7.7821e-01, -1.2331e+00],
                                     [-1.6506e+00, 2.7580e-01],
                                     [1.4286e+00, 5.6542e-01],
                                     [1.6365e+00, -1.2910e+00],
                                     [-1.7541e+00, 6.0856e-01],
                                     [2.2657e-01, -3.8283e-01],
                                     [-1.2291e+00, 1.6032e+00],
                                     [3.0956e-01, -1.4976e+00],
                                     [-4.6846e-01, 2.0793e+00],
                                     [2.0092e+00, -2.1354e-01],
                                     [-4.4899e-01, 1.0783e+00],
                                     [6.9621e-01, 7.3421e-01],
                                     [2.0368e+00, 1.4719e+00],
                                     [2.2885e-02, -2.0144e-01],
                                     [-3.1222e-01, -1.4712e+00],
                                     [7.0015e-01, -1.4974e-01],
                                     [2.8938e-01, 7.8616e-01],
                                     [4.7954e-02, -3.8518e-01]])
                module.weight = nn.Parameter(weight_tensor)
                return module.weight

        if cb_vec_dim == 4:

            if codebook_length == 2:
                weight_tensor = torch.Tensor([[ 0.4448,  0.3618,  0.0475, -0.1372],
            [-0.4153, -0.3457, -0.0450,  0.1284]])
                module.weight = nn.Parameter(weight_tensor)
                return module.weight

            if codebook_length == 4:
                weight_tensor = torch.Tensor([[ 0.5743,  0.4048,  0.4226, -0.3215],
                                        [ 0.1206, -0.2859,  0.0010,  0.6753],
                                        [-0.6645, -0.3315,  0.3708, -0.2389],
                                        [-0.0368,  0.2112, -0.7821, -0.1528]])
                module.weight = nn.Parameter(weight_tensor)
                return module.weight

            if codebook_length == 8:
                    weight_tensor = torch.Tensor([[ 0.0366,  0.0229,  0.5862,  0.6763],
                                        [ 0.7176,  0.1074, -0.5191,  0.3198],
                                        [-0.7639, -0.2151,  0.5546, -0.3805],
                                        [ 0.1539, -0.8475, -0.0118, -0.0018],
                                        [ 0.0364,  0.9781,  0.0012,  0.0106],
                                        [ 0.7352,  0.1414,  0.5984, -0.6497],
                                        [-0.6475, -0.2595, -0.5268,  0.5363],
                                        [-0.1385,  0.1099, -0.7818, -0.6177]])
                    module.weight = nn.Parameter(weight_tensor)
                    return module.weight

            if codebook_length == 16:
                weight_tensor = torch.Tensor([[-0.1893, -0.6480, -0.6644,  0.7720],
                            [ 0.3684,  0.5526,  0.7205,  0.5524],
                            [-0.2168, -0.0776, -0.0826, -1.0413],
                            [ 0.5064, -0.2078, -0.9724, -0.2378],
                            [-0.7988, -0.1532, -0.7199, -0.0177],
                            [-0.5119,  0.7735,  0.5070, -0.2306],
                            [ 0.8019,  0.9241, -0.1384, -0.2592],
                            [ 0.0644,  0.0096,  0.0241, -0.0400],
                            [-0.5164, -0.2736,  0.4543,  0.7870],
                            [-0.0835, -0.2819,  1.1308, -0.3336],
                            [ 0.0329,  0.5828, -0.4358,  0.6755],
                            [ 1.0131,  0.0889,  0.5685, -0.8023],
                            [ 0.8963, -0.3653,  0.0624,  0.5233],
                            [-0.1931,  0.8586, -0.9260, -0.4516],
                            [-1.2151, -0.5704,  0.3453, -0.3093],
                            [ 0.0281, -0.9932,  0.0560, -0.1301]])
                module.weight = nn.Parameter(weight_tensor)
                return  module.weight

            if codebook_length == 32:
                weight_tensor = torch.Tensor([[ 3.4766e-01,  1.2507e-01, -1.2640e-01, -1.0892e+00],
                                    [ 4.4280e-01, -5.6686e-01, -5.1655e-01, -3.0801e-01],
                                    [ 8.0234e-01,  1.1344e+00,  1.4703e-01, -4.5242e-01],
                                    [ 6.4456e-01,  1.4644e-01,  1.4382e+00, -8.2334e-01],
                                    [-1.3255e-01, -1.1917e+00,  9.1186e-02,  1.1516e-01],
                                    [-4.2180e-01, -2.6612e-01,  6.2819e-02, -1.3537e-01],
                                    [ 2.2820e-01, -7.4188e-02,  3.4675e-01,  1.2450e+00],
                                    [-6.4626e-01, -7.0313e-01, -8.1272e-01,  2.1282e-01],
                                    [-1.6567e-03, -4.9864e-01,  9.6238e-01,  3.9864e-01],
                                    [-7.1472e-01,  5.2262e-01, -6.2819e-01,  5.6307e-02],
                                    [ 6.7681e-02, -6.6226e-01,  5.4588e-01, -7.8139e-01],
                                    [ 1.4844e+00,  1.0097e-01,  1.8956e-01, -8.1208e-01],
                                    [ 1.7538e-01, -2.4115e-01, -1.2510e-03,  4.0558e-01],
                                    [-1.5083e+00, -6.8490e-02, -1.1570e-01, -9.6278e-02],
                                    [ 4.9803e-01,  6.2410e-01,  7.5560e-01,  5.0008e-01],
                                    [ 8.6257e-01,  4.4456e-01, -9.7675e-01, -4.7178e-01],
                                    [-1.2146e+00, -9.0961e-01,  4.1216e-01, -5.9676e-01],
                                    [-5.1821e-01, -2.4143e-01, -7.9066e-01, -8.5578e-01],
                                    [-6.5815e-01, -6.9959e-02,  1.0683e+00, -1.8827e-01],
                                    [-5.6038e-01,  3.5698e-01,  3.0715e-01, -8.9963e-01],
                                    [ 1.0023e+00,  2.5509e-01, -3.4431e-01,  4.9724e-01],
                                    [ 9.5611e-02, -3.5877e-02, -1.4383e+00,  1.0160e-01],
                                    [ 4.0207e-01,  1.1499e-01,  4.0900e-01, -2.3194e-01],
                                    [-4.8534e-01,  3.6667e-01,  3.6965e-01,  5.2113e-01],
                                    [-3.4844e-01,  5.7532e-02, -6.1491e-01,  8.8227e-01],
                                    [-9.4168e-01, -6.1781e-01,  2.4008e-01,  8.9495e-01],
                                    [ 3.8137e-01, -8.1724e-01, -6.4828e-01,  8.0509e-01],
                                    [ 1.1028e+00, -7.0402e-01,  3.9883e-01,  1.6849e-01],
                                    [-2.9324e-01,  1.1215e+00,  6.8277e-01, -1.7218e-01],
                                    [ 1.8711e-01,  1.0481e+00, -3.5105e-01,  5.7069e-01],
                                    [ 1.1414e-01,  3.4028e-01, -3.4040e-01, -7.0288e-02],
                                    [-1.6870e-01,  1.1389e+00, -8.8398e-01, -6.6401e-01]])
                module.weight = nn.Parameter(weight_tensor)
                return module.weight

            if codebook_length == 64:
                weight_tensor =  torch.Tensor([[ 6.4920e-01, -3.1010e-01,  6.2549e-01, -1.0665e+00],
            [ 4.7763e-01,  3.7385e-01, -1.0873e-01,  3.2443e-01],
            [ 1.2167e+00,  6.2832e-01, -5.5400e-01, -1.8118e-01],
            [-1.1418e-01,  7.0420e-01,  6.9924e-01, -9.1699e-01],
            [ 5.1607e-01,  4.3191e-01, -1.1177e+00,  5.2929e-01],
            [-2.9593e-02, -7.2515e-01,  3.6572e-03,  3.5699e-02],
            [-6.5193e-01, -1.1999e+00,  4.6905e-01,  4.3234e-01],
            [-1.6659e+00, -7.9005e-01, -1.7008e-02,  4.2612e-01],
            [-3.0110e-01,  1.7153e-03, -5.8787e-01,  2.8705e-01],
            [-7.2972e-01, -2.3149e-01,  4.3349e-01,  1.3626e+00],
            [-5.3647e-01, -1.0888e+00, -1.9842e-01, -7.0260e-01],
            [-3.5278e-01,  1.2687e+00,  1.0216e-01,  6.6138e-01],
            [-8.8278e-01, -4.5431e-01, -6.7992e-01,  1.0395e+00],
            [ 2.8731e-01,  5.9063e-01,  9.2716e-01,  6.9158e-01],
            [ 8.7892e-01,  1.3345e+00, -1.0752e-01, -9.4361e-01],
            [ 9.6461e-01,  7.2119e-01,  1.3245e+00, -1.0289e+00],
            [-1.0830e+00,  3.7303e-01, -1.4535e-01,  4.0863e-01],
            [ 3.5634e-01, -1.0945e+00,  6.5003e-02,  9.1301e-01],
            [ 6.0544e-01, -1.1662e+00, -2.9932e-01, -2.0762e-01],
            [-9.9896e-01, -4.6610e-01, -8.8037e-01, -1.1349e-01],
            [-5.3291e-01,  9.4287e-01,  1.1406e+00,  3.9218e-02],
            [ 1.7523e+00,  1.4618e-01,  1.7449e-01, -9.3781e-01],
            [ 2.0025e-01,  1.8747e-01,  8.0566e-01, -1.8086e-01],
            [-1.1063e-01,  7.4419e-01, -4.1811e-02, -4.6064e-02],
            [ 5.3481e-01,  1.2818e+00,  5.6094e-01, -8.0558e-03],
            [ 1.3739e+00, -8.2515e-01,  6.0785e-01,  6.9784e-02],
            [-1.6267e+00, -1.0946e-01, -1.5615e-01, -7.7092e-01],
            [ 6.8521e-02,  4.0234e-01, -2.9665e-01, -1.0830e+00],
            [-2.5860e-01, -3.7597e-01, -6.5540e-03,  6.9730e-01],
            [ 7.2005e-01, -2.4713e-01, -4.9071e-02, -2.7387e-01],
            [-5.6971e-01, -2.3066e-01,  2.4190e-01, -1.3898e+00],
            [-7.5530e-01, -3.8511e-01, -1.8401e-03, -2.0096e-02],
            [ 3.2740e-02, -4.6142e-01,  1.1615e+00,  6.7021e-01],
            [ 8.2524e-01,  8.7528e-01, -2.3981e-02,  8.5060e-01],
            [-5.4955e-01,  1.3302e+00, -3.2339e-01, -7.1348e-01],
            [ 3.4504e-01,  1.3390e+00, -6.2215e-01,  1.0397e-01],
            [ 2.2685e-01, -4.8874e-01, -9.1166e-01,  1.1014e+00],
            [-1.2660e+00, -8.6641e-01,  6.7877e-01, -5.8520e-01],
            [-2.3800e-01,  2.7842e-01,  3.5610e-01,  5.2725e-01],
            [ 5.8060e-01, -3.0332e-01,  1.7634e+00, -5.9228e-01],
            [-8.7434e-02, -2.6946e-01,  2.3909e-01, -6.4497e-01],
            [ 5.8942e-01,  4.9077e-01,  1.2665e-01, -5.1504e-01],
            [-3.1481e-01, -1.0714e+00, -7.1145e-01,  3.2518e-01],
            [-7.5093e-01,  3.2406e-01,  4.7347e-01, -3.5109e-01],
            [-1.0355e+00, -8.2974e-02,  9.2079e-01,  3.8825e-01],
            [ 2.6668e-01,  3.1469e-01, -6.9475e-01, -2.4793e-01],
            [ 4.6533e-01, -2.9574e-01,  4.0288e-01,  3.6457e-01],
            [ 4.2207e-01, -3.5369e-01, -4.7832e-01,  3.4434e-01],
            [-6.0442e-01,  7.4437e-01, -1.0495e+00,  1.4072e-01],
            [-2.8509e-01, -2.8177e-01, -1.5027e+00,  1.4489e-01],
            [ 8.2134e-01, -2.0813e-01, -6.6538e-01, -1.0740e+00],
            [-5.3566e-01,  1.0946e-02, -1.1986e+00, -1.0481e+00],
            [ 1.0444e+00,  3.3619e-01,  5.9806e-01,  3.3697e-02],
            [-5.2957e-01, -2.5702e-01,  1.2415e+00, -5.9137e-01],
            [ 1.7186e-01, -9.4681e-01,  7.1924e-01, -3.1448e-01],
            [-5.9482e-01,  2.2210e-01, -3.9042e-01, -4.7959e-01],
            [-3.3559e-04, -4.4204e-01, -6.3583e-01, -4.5349e-01],
            [ 1.4455e+00, -2.2919e-01, -3.3319e-01,  6.9005e-01],
            [-1.0547e-02,  3.1853e-02,  8.2748e-03, -7.1739e-02],
            [-3.4747e-01, -3.2886e-01,  6.4349e-01,  4.5740e-02],
            [ 4.1281e-01, -1.9055e-02,  2.7358e-01,  1.2257e+00],
            [ 7.1171e-01, -5.3636e-01, -1.3500e+00, -8.9763e-02],
            [-1.2277e-01,  4.9499e-01, -4.2825e-01,  9.2991e-01],
            [ 2.6475e-01,  7.9880e-01, -1.5477e+00, -6.4908e-01]])
                module.weight = nn.Parameter(weight_tensor)
                return module.weight

            if codebook_length == 128:
                weight_tensor =  torch.Tensor([[ 1.1106e-01, -3.6073e-01,  4.5174e-01,  8.4666e-01],
                    [-4.2661e-02, -5.9570e-01,  5.5363e-01,  2.4088e-01],
                    [ 7.0835e-01, -1.3596e+00, -3.3809e-01,  1.1912e-01],
                    [ 6.2797e-01,  1.1991e+00, -7.0006e-01,  6.4414e-01],
                    [-9.7075e-01, -6.6466e-01, -7.0975e-01, -5.5608e-01],
                    [-2.0644e-01, -1.5514e+00,  2.7272e-01,  4.6519e-02],
                    [ 3.1093e-01, -3.2331e-01, -3.2083e-01, -2.3503e-01],
                    [ 5.7409e-01, -5.1720e-01, -6.1344e-01,  2.8842e-01],
                    [ 7.6914e-02,  2.1020e-01, -4.4357e-01,  7.9856e-03],
                    [-1.2452e+00, -6.0064e-02,  8.0572e-01,  4.4130e-01],
                    [-7.1941e-01,  2.8149e-02, -1.1547e+00,  7.3443e-01],
                    [ 1.9936e-01, -3.1473e-01, -1.1986e+00,  1.1180e+00],
                    [ 1.6294e+00, -8.6449e-01,  5.6537e-01,  1.0944e-01],
                    [ 5.6739e-01, -1.1260e+00,  7.9163e-01,  2.0855e-01],
                    [-7.0131e-01, -6.7028e-01,  1.7068e-01,  7.0099e-01],
                    [-7.9118e-01, -2.9340e-01,  9.6054e-02, -6.6175e-01],
                    [-5.7488e-01,  8.7937e-01, -1.5822e+00, -7.2066e-01],
                    [ 2.7614e-01,  3.4444e-01,  1.2237e-01, -1.1549e-01],
                    [ 2.2039e-01,  3.4911e-01,  9.9554e-01,  7.2260e-01],
                    [ 1.4870e-01, -4.3720e-01,  9.5457e-01, -3.3943e-01],
                    [-9.9387e-01, -1.4880e+00, -3.9684e-01,  2.2582e-01],
                    [-3.6911e-01,  2.3841e-01,  1.0163e-01,  5.2283e-02],
                    [-1.6135e-01,  2.2300e-01, -1.0856e-01, -5.4337e-01],
                    [ 1.4833e+00,  6.4544e-01, -5.1664e-01,  9.1499e-02],
                    [-1.6013e+00,  2.1705e-01,  2.3158e-01, -2.9998e-01],
                    [-3.7033e-01, -2.5882e-01, -3.7863e-01, -1.8867e-01],
                    [ 3.6677e-01,  1.6903e+00,  4.3463e-01, -1.1730e-01],
                    [ 6.0298e-01,  8.1919e-01, -1.6963e+00, -3.0477e-01],
                    [ 7.5523e-01, -8.9399e-01,  5.6350e-01, -8.3929e-01],
                    [-1.5006e-01, -2.2305e-01, -6.2270e-01, -9.0323e-01],
                    [-1.3037e+00,  1.4440e-01, -8.9364e-01, -1.3898e-01],
                    [-5.6893e-01,  1.4578e+00, -2.4233e-01, -8.6857e-01],
                    [ 6.3409e-03,  7.6364e-02,  6.5784e-01,  5.5440e-02],
                    [ 8.2017e-01, -1.3802e-01,  6.2750e-01,  1.2301e+00],
                    [ 1.1226e+00,  9.1617e-02, -9.6011e-01, -8.8104e-01],
                    [ 2.6180e-01,  7.3515e-01, -8.6853e-01, -1.2317e+00],
                    [-1.1519e-03, -8.0422e-01, -1.8142e-01,  4.4886e-01],
                    [ 9.8384e-01,  1.2358e-01, -1.1649e+00,  4.9662e-01],
                    [-1.8303e+00, -6.0612e-01, -1.3059e-01, -9.6668e-01],
                    [ 8.4984e-02, -5.3267e-01,  1.3597e+00,  6.1839e-01],
                    [-6.0302e-01, -7.4046e-01, -1.5252e+00,  7.6210e-02],
                    [-1.7625e+00, -5.4101e-01, -3.0665e-01,  2.8715e-01],
                    [ 9.7039e-01,  4.3888e-01,  3.2582e-02, -5.5128e-01],
                    [ 1.1058e+00,  7.1978e-01,  1.5557e+00, -1.1863e+00],
                    [-3.6497e-01, -1.9419e-01, -3.2211e-01,  6.9786e-01],
                    [ 2.2154e-01,  3.8994e-01, -8.1776e-01,  4.9304e-01],
                    [ 2.6139e-01, -8.5277e-01,  7.9159e-02, -2.2746e-01],
                    [-9.5074e-01, -7.1996e-01, -6.5206e-01,  1.2989e+00],
                    [ 1.6727e+00, -2.8969e-01, -3.0223e-01,  7.3707e-01],
                    [-3.5450e-01,  9.4693e-01, -1.2051e+00,  3.5027e-01],
                    [-1.7997e-01,  1.4703e+00,  3.3826e-03,  6.8995e-01],
                    [-9.6937e-01,  2.2886e-01, -7.0059e-01, -1.3076e+00],
                    [-4.1309e-01, -5.3042e-01,  8.8970e-02, -1.5542e+00],
                    [ 3.3891e-01, -1.2453e-01, -2.6466e-01,  7.7013e-01],
                    [ 1.3429e+00,  1.1422e+00,  4.5373e-01, -2.5680e-01],
                    [ 5.8898e-01,  6.7422e-01, -7.1150e-01, -1.7828e-01],
                    [ 2.5083e-02,  7.7433e-01,  5.3693e-01,  2.7386e-01],
                    [-6.7098e-01, -9.6006e-01,  9.7565e-01,  3.2635e-01],
                    [-5.9013e-02, -2.2624e-01, -1.6399e+00, -9.5991e-01],
                    [-3.8185e-01, -1.1508e+00, -1.8218e-01, -7.7253e-01],
                    [ 6.7648e-02,  4.7472e-01, -7.0077e-02,  5.6320e-01],
                    [ 7.4723e-01, -1.4678e-01,  1.0389e+00,  1.6664e-01],
                    [-5.8928e-01, -2.5906e-01,  7.8642e-01,  1.3962e+00],
                    [ 2.5257e+00, -6.3721e-01,  4.5986e-01, -2.0711e+00],
                    [ 9.9593e-02,  1.1785e+00,  1.3569e+00,  4.9720e-01],
                    [-1.2063e-01, -2.7729e-01,  2.3463e-01, -2.7202e-01],
                    [-3.3634e-01, -8.9518e-01,  6.8545e-01, -6.4512e-01],
                    [ 3.2163e-02,  4.0402e-01,  1.3062e+00, -2.7682e-01],
                    [-1.2079e-01, -1.0811e+00, -8.4520e-01,  7.0369e-01],
                    [-2.5388e-02,  1.0695e+00,  1.0389e+00, -8.7901e-01],
                    [ 1.1592e-02, -1.2310e+00,  2.9210e-01,  1.1118e+00],
                    [ 7.2218e-01, -8.2464e-01, -1.5072e+00,  2.0186e-02],
                    [ 5.8645e-01, -8.1763e-01, -6.2368e-01, -8.9959e-01],
                    [ 1.7525e-02,  5.7726e-02, -1.6947e+00,  1.8867e-01],
                    [ 1.2633e+00,  1.3402e+00, -5.7677e-01, -6.9328e-01],
                    [-9.3477e-01, -1.1771e-01, -3.6362e-02,  1.3907e-01],
                    [-7.6550e-01, -5.7212e-01, -6.3744e-01,  3.0748e-01],
                    [-7.2023e-01,  6.5845e-01,  2.9753e-01,  5.7210e-01],
                    [-3.2813e-01,  8.5560e-01,  3.4247e-01, -3.6874e-01],
                    [ 8.1918e-01,  1.0134e+00,  5.3584e-01,  7.5922e-01],
                    [-9.8905e-01,  9.9438e-01, -4.2680e-01,  1.6645e-01],
                    [ 8.0516e-01, -5.8740e-02,  7.8053e-03, -1.4033e+00],
                    [ 4.6258e-01,  1.7717e-01, -3.3538e-01, -6.5303e-01],
                    [ 1.4808e-01,  1.7189e+00, -7.4142e-01, -2.3184e-01],
                    [ 6.1478e-01, -1.9499e-01,  3.2079e-01, -2.4890e-01],
                    [ 6.8778e-01, -5.1723e-01,  1.6291e-01,  3.6184e-01],
                    [ 1.0730e+00, -3.9552e-01, -2.8064e-01, -3.2938e-01],
                    [-1.6369e-01,  6.1346e-01, -7.9837e-01, -4.3690e-01],
                    [ 3.4263e-01,  3.4243e-01,  5.3754e-01, -7.2161e-01],
                    [ 7.9957e-01,  1.1958e+00,  2.7679e-01, -1.2335e+00],
                    [-7.4938e-02, -2.3816e-01, -1.2162e-01,  1.4742e+00],
                    [-7.4002e-01,  3.5751e-01,  6.9897e-01, -1.6393e-01],
                    [-6.6902e-01, -2.8911e-01,  1.3492e+00, -4.8388e-01],
                    [ 6.6675e-01,  6.8381e-01,  7.8053e-01, -1.2958e-01],
                    [-7.8335e-01,  4.3527e-01, -2.6420e-01, -4.8093e-01],
                    [ 4.9468e-01,  1.8537e-01,  3.5833e-01,  4.7022e-01],
                    [-9.8569e-01,  3.4690e-01,  6.3316e-01, -1.1469e+00],
                    [-1.4147e+00, -7.9777e-01,  9.3709e-01, -8.7803e-01],
                    [-9.0563e-02,  8.9264e-01, -2.9955e-01,  8.4947e-02],
                    [ 1.8191e+00,  1.7748e-01,  1.4786e-01, -8.6694e-01],
                    [-1.2336e+00, -9.6042e-01,  4.0204e-01, -2.2967e-01],
                    [ 6.5911e-01,  8.5224e-01, -3.2514e-02,  2.0746e-01],
                    [-4.7980e-01, -2.1292e-02, -1.0410e+00, -2.6619e-01],
                    [ 1.5974e-02,  5.1985e-01,  4.2311e-01,  1.2119e+00],
                    [ 7.1849e-01,  1.5328e-01, -2.8551e-01,  1.6608e-01],
                    [ 6.7319e-01, -8.4011e-01, -3.3556e-01,  1.1111e+00],
                    [-3.4877e-01,  3.2570e-02,  3.7957e-01,  6.2264e-01],
                    [ 5.1611e-02, -2.0098e-01, -4.7019e-03,  2.5089e-01],
                    [-5.0388e-01,  2.9435e-01, -5.3743e-01,  2.8402e-01],
                    [ 3.3750e-01,  9.6418e-01, -6.3050e-02, -5.2706e-01],
                    [-2.4729e-01,  6.2866e-01, -5.0657e-01,  1.0823e+00],
                    [ 5.9317e-01, -3.5930e-01,  1.9339e+00, -6.8406e-01],
                    [-1.6389e+00, -9.9622e-01,  5.9650e-01,  1.0521e+00],
                    [ 3.7944e-01, -7.2151e-02, -9.8545e-01, -2.8389e-01],
                    [ 1.1978e+00,  1.8699e-01,  3.8429e-01,  1.1203e-01],
                    [-6.0764e-02, -3.0483e-01, -8.2729e-01,  2.9333e-01],
                    [-1.4218e-01,  4.8056e-01,  5.6602e-02, -1.2399e+00],
                    [ 1.4236e-01, -3.6139e-01,  1.4175e-01, -8.4295e-01],
                    [-6.5763e-02, -8.9977e-01, -7.5018e-01, -2.1055e-01],
                    [-8.9072e-01,  1.1298e+00,  1.0707e+00,  2.8801e-02],
                    [ 9.8663e-01, -2.8843e-02,  8.8343e-01, -8.0632e-01],
                    [-3.2161e-01,  6.1367e-02,  6.1497e-01, -6.1871e-01],
                    [-4.7493e-01, -7.4133e-01, -2.7999e-03, -2.2299e-02],
                    [ 8.0232e-01,  4.4631e-01, -2.3017e-01,  1.0843e+00],
                    [-5.2811e-01,  5.7152e-02,  1.2062e+00,  3.8115e-01],
                    [-6.3165e-01, -3.2801e-01,  6.0400e-01, -2.8979e-02],
                    [ 1.3595e-02, -1.8214e-01,  1.0392e+00, -1.2454e+00],
                    [-1.1189e+00,  1.1630e-01, -1.2787e-01,  1.0226e+00]])
                module.weight = nn.Parameter(weight_tensor)
                return module.weight

            if codebook_length == 256:
                weight_tensor = torch.Tensor([[ 1.2408e+00, -4.1544e-01, -8.5814e-01, -1.3973e+00],
                [ 1.9417e-01, -1.8511e-01,  5.1801e-02,  2.9974e-01],
                [-6.4737e-01,  6.7125e-01,  2.2945e-01,  1.3695e+00],
                [ 5.0151e-01, -7.1363e-01, -7.7080e-01, -3.7824e-02],
                [ 9.8562e-01,  2.6009e-01,  7.1851e-01,  2.6068e-01],
                [ 1.9270e-01,  1.1629e+00, -8.8552e-01,  7.2475e-02],
                [-8.4419e-01,  8.5994e-01, -6.5854e-03, -2.5147e-01],
                [-3.8127e-01,  6.3866e-01, -9.9388e-01, -7.3125e-01],
                [ 1.8362e+00,  2.4914e-01, -6.1831e-01,  9.2708e-01],
                [ 6.0719e-01,  8.1215e-01, -2.8717e-01,  4.4199e-01],
                [-6.5980e-01, -7.4156e-01,  6.4734e-01,  1.4294e-01],
                [-1.8135e+00, -6.0042e-01,  6.5763e-01, -1.1736e+00],
                [ 1.6170e+00, -4.1266e-01,  6.1225e-01, -6.4621e-02],
                [ 9.3792e-01,  1.0702e+00,  6.8999e-01,  6.7344e-01],
                [-1.3905e+00, -9.2123e-01,  1.0608e+00, -5.5609e-01],
                [-2.2855e-01,  6.3948e-01, -8.2892e-02,  5.6970e-01],
                [ 1.6914e+00,  3.3073e-01, -6.7823e-01, -3.6035e-01],
                [ 3.0732e-01,  1.6876e+00,  1.0737e+00,  1.5900e-01],
                [-6.4570e-02,  2.0322e-01,  9.3206e-02, -5.2018e-01],
                [ 1.9810e+00,  1.3293e-01,  1.9421e-01, -9.0004e-01],
                [ 3.6088e-01, -3.2801e-01, -1.8886e+00, -7.6760e-01],
                [-5.0046e-01,  2.2944e-01,  1.0854e+00, -1.3916e+00],
                [ 1.9289e-01,  1.1807e-01, -1.7046e+00,  2.7900e-01],
                [-1.1414e-01,  7.1825e-02,  1.1943e+00, -5.2604e-01],
                [ 4.7397e-01,  1.1240e+00, -1.7333e-01, -2.1457e-01],
                [-4.2757e-02,  1.6290e-01, -3.4959e-01,  1.6284e-01],
                [-1.1079e+00,  3.0615e-01, -1.1249e+00, -1.3748e+00],
                [-6.2849e-01,  1.0084e-01, -6.5571e-01, -4.5259e-01],
                [ 6.3733e-01,  2.6623e-01, -3.3863e-01,  9.6725e-02],
                [ 1.6972e-01,  7.1352e-01,  1.0102e+00,  6.4111e-01],
                [ 3.5635e-01, -1.5204e+00,  5.4057e-02, -6.7161e-01],
                [-1.3575e+00, -1.9620e-02,  7.6509e-02,  1.0664e+00],
                [ 2.6578e-01, -4.1134e-02, -7.2852e-01, -9.7676e-02],
                [ 1.0937e-01, -6.7841e-01,  1.3352e+00, -5.8960e-01],
                [-1.2591e+00, -4.6609e-01, -3.8388e-01, -2.0053e-01],
                [-2.3580e-01,  1.1100e+00, -4.5057e-01, -3.9632e-01],
                [ 5.5065e-01, -9.3422e-01,  1.4561e+00,  4.4088e-01],
                [-8.7633e-02, -5.1502e-01, -5.9251e-01,  1.0254e+00],
                [ 3.9743e-01, -1.6974e+00,  1.0274e-01,  6.8551e-01],
                [ 1.5730e-01,  3.8114e-01,  6.7542e-01,  1.3619e+00],
                [ 5.2855e-01, -7.7594e-01, -1.1129e-01, -4.4491e-01],
                [-1.4854e+00,  4.5782e-01,  5.8763e-01, -4.7091e-01],
                [-1.5502e+00,  5.5568e-01, -3.4566e-01,  2.3812e-01],
                [ 1.1511e+00, -5.4645e-01, -7.7955e-01,  5.1471e-01],
                [-3.0518e-02, -1.0810e+00, -6.7534e-02,  1.3103e+00],
                [-5.2118e-01, -1.3952e-04,  1.1218e+00,  1.0724e+00],
                [-6.7251e-01,  1.4540e-01,  1.4560e-01, -5.4811e-01],
                [-2.0002e-01, -4.9255e-01,  3.3999e-01,  2.6869e-01],
                [ 5.7562e-01, -2.8256e-01,  2.0931e+00, -7.8361e-01],
                [-1.2051e-01,  1.5392e+00,  3.1528e-01, -3.3902e-01],
                [-5.3749e-01,  7.1044e-01,  5.2964e-01, -7.7413e-01],
                [-6.8498e-01, -4.1542e-01,  1.4366e+00, -8.1584e-01],
                [-3.4920e-01, -1.2371e+00, -7.6751e-01,  6.7723e-01],
                [ 4.4729e-01,  9.0365e-01,  4.1104e-01,  2.1972e-01],
                [ 9.4306e-01, -8.9663e-01,  8.0710e-01, -9.5109e-01],
                [ 4.1350e-01, -2.8049e-01, -9.4497e-01,  4.8265e-01],
                [-8.8021e-01,  1.5251e+00, -3.0464e-01, -9.2338e-01],
                [ 1.1456e-01,  4.2979e-01, -4.5897e-01, -4.4600e-01],
                [-7.0637e-01, -4.0900e-01,  7.2636e-02,  1.1599e-01],
                [-5.2273e-01, -7.7786e-01,  5.0641e-01,  9.0524e-01],
                [-3.9372e-02, -6.4416e-01,  8.3465e-01, -1.3242e+00],
                [-7.0425e-01, -5.4528e-01, -9.6006e-01,  4.6722e-01],
                [-6.4267e-01, -4.5491e-01, -2.2915e-01, -4.6850e-01],
                [ 8.8865e-01, -2.5813e-01, -4.6750e-01, -1.6393e-01],
                [-6.8537e-01, -1.3652e+00,  4.7868e-01, -1.1413e+00],
                [ 9.8127e-01,  2.8860e-01, -6.7647e-01,  5.2148e-01],
                [ 1.1424e-01,  2.0640e+00, -5.5066e-01, -1.9964e-01],
                [-2.7803e-02,  2.9257e-01, -2.0471e-01,  1.1314e+00],
                [ 1.0900e+00,  1.4958e+00, -8.2987e-01, -3.8417e-01],
                [ 6.4369e-01, -3.7531e-01,  5.6343e-01, -4.5822e-01],
                [ 1.6805e-01,  8.6241e-01,  7.1875e-02, -8.8885e-01],
                [-1.9235e+00, -9.3762e-01,  8.0005e-01,  1.0416e+00],
                [-5.5178e-01,  1.7977e+00, -3.1185e-01,  4.9184e-01],
                [-1.4345e-01, -1.8691e-01, -1.0430e-01, -8.6414e-01],
                [-1.1857e+00, -1.1551e+00, -8.9780e-01,  9.6374e-02],
                [ 3.5766e-01,  9.4828e-01, -7.5309e-01, -7.7846e-01],
                [ 5.2893e-01, -5.6630e-02,  1.0140e+00, -1.0653e+00],
                [-5.0703e-01,  3.7463e-01,  2.2065e-01,  1.4606e-01],
                [-4.9231e-01, -7.5573e-01,  7.6733e-01, -5.2238e-01],
                [-2.1337e-01,  8.8649e-01,  5.7208e-01,  2.2931e-01],
                [ 3.5263e-01, -3.8144e-01, -3.2183e-01,  2.9507e-02],
                [-1.0596e+00, -9.3304e-01,  3.0328e-01, -3.9150e-01],
                [ 3.2310e-01,  3.1243e-01, -2.4569e-01,  5.7961e-01],
                [ 4.3724e-01, -6.8005e-02,  1.1756e+00,  8.3089e-01],
                [-1.0637e+00, -7.5514e-01, -8.9223e-01,  1.3033e+00],
                [ 9.5807e-01,  3.8940e-01,  6.9972e-02,  7.1547e-01],
                [ 4.8684e-01, -5.6869e-01, -9.1293e-01,  1.6319e+00],
                [-1.2382e+00,  2.8455e-01, -4.1243e-01, -5.9767e-01],
                [ 6.1201e-01, -3.1325e-01, -8.2555e-01, -7.0579e-01],
                [ 8.6995e-01,  4.3104e-02,  1.1037e-01, -3.9078e-01],
                [ 1.1460e+00,  1.8319e-01,  6.3533e-01,  1.2553e+00],
                [ 8.2684e-02, -4.7410e-01,  5.2027e-02,  8.0787e-01],
                [ 7.9062e-02,  3.2773e-01, -9.8633e-01,  1.0048e+00],
                [ 9.4431e-01,  7.0320e-01,  1.3552e-01,  1.0494e-02],
                [-1.1815e+00,  3.3727e-01,  1.4491e-01, -1.3731e+00],
                [-6.3935e-01, -1.1337e+00, -4.5848e-01, -6.7523e-01],
                [ 2.7363e-01, -8.6591e-01,  5.5740e-01,  7.1128e-01],
                [ 7.0577e-01,  6.7891e-01, -2.5468e-01,  1.3435e+00],
                [-4.3536e-01,  4.9763e-01, -2.3602e-01, -8.4008e-01],
                [-6.1037e-01, -1.7814e+00, -1.6684e-01, -3.8366e-02],
                [ 9.0160e-02,  4.3139e-01,  1.4210e-01,  1.7471e-01],
                [-1.1255e+00, -2.1780e-01, -4.7859e-01,  4.7976e-01],
                [-7.5570e-01,  1.3338e-01, -1.5119e+00,  7.0168e-01],
                [-3.3500e-01,  5.4375e-01, -2.1543e+00, -6.1047e-01],
                [ 1.1881e+00, -1.0483e-02,  7.1368e-01, -9.6410e-01],
                [-5.8737e-01,  9.4545e-01, -5.5269e-01,  2.5061e-01],
                [ 2.5106e-01,  1.9161e-01, -4.5828e-01, -1.1026e+00],
                [ 1.6872e+00, -1.2745e+00,  5.6704e-01,  2.7320e-01],
                [ 6.0020e-01, -1.2900e-01, -4.5052e-01,  1.0230e+00],
                [-7.9765e-01, -6.9453e-01, -2.4951e-01,  9.6093e-01],
                [-3.4924e-01,  1.0441e+00,  1.6702e+00,  6.6823e-01],
                [ 2.1497e-01, -2.2508e-01,  9.8380e-02,  1.6119e+00],
                [-3.8125e-01,  4.0664e-01,  5.3863e-01,  7.2185e-01],
                [ 7.9976e-01, -8.4656e-01, -6.6365e-02,  1.1242e+00],
                [-7.3197e-01, -3.5233e-01,  3.5080e-01, -1.0541e+00],
                [ 5.1368e-01, -7.1735e-02,  2.3836e-01,  9.1418e-01],
                [-6.4489e-01, -8.8955e-01, -3.8988e-01,  1.0918e-01],
                [-1.8010e-01, -3.3865e-01,  9.4064e-01,  4.8141e-01],
                [ 1.7180e-01, -8.3234e-01,  5.3866e-01, -5.1648e-01],
                [ 1.3671e+00,  5.0308e-01,  2.4413e-01, -4.9960e-01],
                [-9.5049e-02, -6.5022e-01, -7.7266e-01,  2.7470e-01],
                [ 1.8162e+00,  1.4195e+00, -8.0456e-02, -1.1178e+00],
                [ 6.9962e-01, -2.7532e-01, -1.7516e-01,  4.6376e-01],
                [ 2.6256e-01,  4.2044e-01,  7.6616e-01,  3.3165e-02],
                [ 9.1576e-01, -4.6379e-01,  5.8339e-01,  5.4301e-01],
                [-5.3296e-01, -6.3099e-02, -6.4351e-01,  8.3603e-01],
                [ 3.3293e-01,  2.7535e-01,  6.0873e-01,  6.1225e-01],
                [ 1.2688e+00,  7.8482e-02, -3.3782e-02,  1.1861e-01],
                 [-6.7615e-01, 2.2308e-01, -1.5561e-01, 5.5382e-01],
                 [4.3304e-01, -1.7641e-01, 1.0515e+00, -2.1856e-02],
                 [-4.3183e-02, -1.8286e-01, 5.3408e-01, 9.1859e-01],
                 [9.1193e-01, -6.5252e-02, -1.2369e+00, -1.0622e-01],
                 [-1.1728e-01, 6.1722e-02, 5.2141e-01, 2.6516e-01],
                 [2.9312e-01, -7.7278e-01, -3.2860e-01, 5.4059e-01],
                 [-6.1207e-01, -1.6290e-01, -4.9686e-01, -1.1259e+00],
                 [-2.5788e-01, 2.3957e-01, 5.6919e-01, -2.7794e-01],
                 [-7.0200e-01, -3.4511e-01, 1.2701e+00, -9.6247e-02],
                 [7.4871e-01, -9.3653e-01, -1.7964e+00, 1.0842e-01],
                 [-6.5248e-01, -2.5291e-02, -5.5772e-01, 1.7943e+00],
                 [1.7261e+00, -4.6050e-01, -1.1206e-01, 7.1196e-01],
                 [-8.1416e-01, 2.8042e-01, -7.8614e-01, 1.6102e-01],
                 [5.7434e-01, 2.0169e-01, 2.2078e-01, 2.1871e-01],
                 [5.4068e-01, 2.1136e-01, 2.1503e-01, -8.9409e-01],
                 [2.6800e-01, -7.1448e-01, 9.5742e-01, 1.5123e+00],
                 [8.7202e-01, 7.0629e-01, -7.2830e-01, -1.0379e-01],
                 [-5.0155e-01, 1.1241e-01, -1.4073e+00, -2.8383e-01],
                 [1.1996e+00, 1.4058e-01, -2.1501e-01, -1.0427e+00],
                 [-9.0225e-01, 9.5135e-01, 2.5846e-01, 5.2388e-01],
                 [9.8601e-01, 4.8732e-01, -1.1248e+00, -9.1451e-01],
                 [-3.0057e-01, 8.7150e-01, -4.9478e-01, -1.4853e+00],
                 [-1.2916e-01, 8.0688e-02, 8.3084e-02, 6.1519e-01],
                 [1.3953e+00, 8.0019e-01, 1.0397e+00, -4.1192e-01],
                 [3.9086e-01, -1.9458e-01, -2.1662e-01, -5.6857e-01],
                 [-2.0418e-01, -6.2477e-02, -1.0196e+00, 2.6366e-01],
                 [7.8110e-01, 2.8769e-01, 7.1063e-01, -3.7185e-01],
                 [-4.1847e-01, -3.6452e-01, 2.7986e-01, -3.5665e-01],
                 [7.5963e-01, -1.0562e+00, 6.5973e-01, -6.7151e-02],
                 [-1.1276e+00, -1.3356e+00, 7.0971e-01, 1.9804e-01],
                 [-8.1545e-02, -9.4930e-02, -8.2936e-01, -6.7984e-01],
                 [8.9513e-01, -9.7359e-01, -1.0007e-01, 2.1246e-01],
                 [-9.7825e-01, 9.5699e-01, -1.0943e+00, -2.5778e-01],
                 [4.4993e-02, 2.2075e-01, -1.2996e+00, -1.3670e+00],
                 [9.9509e-01, 1.5987e+00, 3.4660e-01, -2.2617e-01],
                 [8.0676e-02, -7.8699e-01, -7.5676e-01, -1.1862e+00],
                 [7.2733e-01, -3.3030e-01, 1.8507e-01, -1.2431e-02],
                 [-7.4206e-02, -1.9593e-01, 1.6981e+00, 2.4175e-01],
                 [-4.9961e-01, -1.5107e-01, 9.1612e-02, 1.1713e+00],
                 [8.1216e-01, 8.8372e-01, -1.8368e+00, -1.2259e-01],
                 [-8.4781e-01, 6.0558e-02, -1.5423e-01, -8.1686e-02],
                 [-5.3488e-01, -9.4407e-01, 1.2814e+00, 6.9015e-01],
                 [-8.4302e-01, 6.6453e-01, -6.8595e-01, 9.5844e-01],
                 [-3.6999e-03, 1.3674e+00, 4.9841e-01, 9.7667e-01],
                 [-7.4167e-02, 1.1271e+00, -6.4083e-02, 2.5922e-01],
                 [-1.4555e-01, 9.0673e-01, 1.0924e+00, -2.3806e-01],
                 [2.7071e-01, 1.5668e-02, 4.0589e-01, -2.3373e-01],
                 [1.5290e+00, 1.0613e+00, -1.5719e-01, 2.3109e-01],
                 [8.5269e-02, -4.0559e-01, 1.3713e-01, -2.7985e-01],
                 [-3.9566e-01, 4.5741e-01, -2.7319e-01, -1.1918e-01],
                 [1.1373e+00, -2.5336e-01, 5.3340e-01, -1.9694e+00],
                 [-2.1700e+00, -1.7947e-01, -3.9707e-01, -8.0800e-01],
                 [2.9916e-01, 4.1774e-01, -1.2156e+00, -4.1044e-01],
                 [-1.1077e+00, -1.5565e+00, -4.7356e-02, 9.5001e-01],
                 [5.9383e-01, 1.5662e+00, -1.1598e-01, 5.3807e-01],
                 [7.8272e-01, 1.1292e+00, -1.0040e+00, 7.1724e-01],
                 [2.4495e-03, -7.7513e-01, 8.4954e-01, 7.0639e-02],
                 [9.1694e-01, 5.6618e-02, -1.5053e+00, 9.0725e-01],
                 [-7.1646e-02, 1.2927e+00, 9.4211e-01, -1.1206e+00],
                 [-1.1881e+00, -7.1943e-01, 1.3916e-01, 4.4199e-01],
                 [-1.9289e+00, -3.3204e-01, 3.2486e-01, -1.1105e-02],
                 [-1.9370e-01, 4.8448e-01, -8.8437e-01, -8.0334e-02],
                 [2.8327e-01, 7.8963e-02, -1.1695e-01, -1.4175e-01],
                 [-1.7627e+00, -1.3797e+00, -8.1405e-02, -3.5075e-01],
                 [-1.1777e+00, 4.2483e-01, 1.1017e+00, 5.4256e-01],
                 [1.0134e+00, -1.2354e+00, -8.1461e-01, -4.1884e-01],
                 [8.6556e-01, 8.7208e-01, 6.2920e-01, -1.1983e+00],
                 [-1.1615e+00, 1.3431e-01, 2.6186e-01, 2.6262e-01],
                 [-3.2677e-01, 2.4872e-01, 1.0868e+00, 2.1880e-01],
                 [4.0457e-01, 6.8307e-01, 1.4237e+00, -7.5065e-01],
                 [-4.5968e-02, 2.8372e-01, 3.2454e-01, -1.3583e+00],
                 [-5.5575e-01, -4.7244e-01, -8.8077e-01, -2.1092e-01],
                 [5.8431e-01, 9.8624e-01, 6.4649e-01, -3.8159e-01],
                 [-2.2360e-01, 4.1007e-01, -6.3971e-01, 4.7794e-01],
                 [1.4174e-01, -1.2056e+00, -3.1068e-01, 4.1986e-02],
                 [-5.9299e-02, 1.4576e+00, -1.4798e+00, -9.0556e-01],
                 [-2.8379e-01, -9.6774e-01, 1.3037e-01, -2.1610e-01],
                 [5.0609e-03, 1.0315e+00, -4.9993e-01, 9.1467e-01],
                 [-2.0086e-02, -6.6858e-01, -5.1437e-01, -4.2720e-01],
                 [2.8465e-01, -7.9485e-01, 2.0421e-01, 1.4073e-01],
                 [5.4220e-01, -4.8560e-01, 7.6855e-02, -1.1686e+00],
                 [-1.7138e-01, -1.0597e-01, -3.6754e-01, -2.8883e-01],
                 [-5.1934e-01, -7.0573e-01, -1.7005e+00, 6.5277e-02],
                 [-7.6416e-01, -5.7464e-01, -1.3102e+00, -8.8805e-01],
                 [7.0541e-02, -1.6056e-01, -4.5438e-01, 5.3506e-01],
                 [7.0382e-01, 3.1728e-01, -4.5595e-01, -5.0251e-01],
                 [2.7155e-01, 6.7813e-01, 2.2202e-01, 7.9559e-01],
                 [-8.9742e-01, 1.4329e+00, 1.0265e+00, -6.4296e-02],
                 [4.6511e-01, 1.7010e+00, -3.2027e-02, -1.0986e+00],
                 [-1.1601e+00, -2.8143e-01, 8.8524e-01, 3.6921e-01],
                 [-6.6861e-01, -1.6463e-01, 4.1540e-01, 5.9899e-01],
                 [-9.8636e-02, -2.7794e-01, 7.3724e-01, -1.9091e-01],
                 [-6.4552e-01, -1.0709e-01, 6.2617e-01, 1.7231e-03],
                 [-8.5620e-02, -5.7231e-01, -1.4469e+00, 9.1475e-01],
                 [-8.3630e-01, 4.6877e-01, 1.4288e+00, -4.3688e-01],
                 [5.8544e-01, 5.8160e-01, 1.4116e+00, 1.4334e-01],
                 [-8.2984e-02, -1.2429e+00, -1.0388e+00, -2.2242e-01],
                 [-3.1426e-01, -1.0696e+00, 6.5957e-02, 4.8255e-01],
                 [9.4146e-01, 8.9605e-01, -1.5038e-01, -7.2608e-01],
                 [4.5597e-01, 5.3641e-01, 1.0866e-01, -3.1337e-01],
                 [7.6327e-01, 6.7334e-01, -3.5636e-01, -1.6608e+00],
                 [3.1378e+00, -7.6632e-01, 4.3494e-01, -1.9888e+00],
                 [3.5508e-01, -2.8858e-01, 5.5234e-01, 3.0840e-01],
                 [-1.1923e+00, -2.8468e-01, 3.5468e-01, -4.1171e-01],
                 [-3.1221e-01, -4.0974e-01, -6.1880e-02, -1.8236e+00],
                 [-3.6379e-03, -1.9403e-01, 5.3241e-01, -7.7902e-01],
                 [-1.0301e-01, 7.5952e-01, 1.9506e-01, -2.7388e-01],
                 [-1.3332e+00, -1.5748e-01, -1.1966e+00, -1.6764e-01],
                 [-1.4584e-01, -5.5577e-01, -2.1510e-01, 7.6410e-02],
                 [1.2713e+00, -5.9932e-01, -5.5051e-02, -5.9348e-01],
                 [5.0900e-01, -1.1894e+00, -9.1021e-01, 6.7303e-01],
                 [1.0158e-01, -4.1989e-01, -1.2472e+00, -2.0119e-01],
                 [3.5899e-01, 4.3801e-01, -9.0237e-01, 3.1898e-01],
                 [-6.9367e-01, -4.8301e-02, 8.0216e-01, -6.1117e-01],
                 [-1.3383e+00, -6.8428e-01, -2.6695e-01, -1.0402e+00],
                 [1.1617e+00, 7.2645e-01, 1.6984e+00, -1.3327e+00],
                 [-3.1308e-01, 1.0379e+00, -1.4548e+00, 3.9710e-01],
                 [-1.4532e-01, -1.5243e+00, 7.1190e-01, 1.5877e-02],
                 [1.4907e-01, 6.7565e-01, -3.7786e-01, 6.7814e-02],
                 [-4.1536e-01, -4.1641e-01, -1.8668e-01, 5.6151e-01],
                 [-1.9959e+00, -5.7349e-01, -5.3229e-01, 5.3246e-01],
                 [1.5093e-01, 4.5315e-01, 6.8001e-01, -6.8874e-01],
                 [-2.5596e-01, -4.5583e-02, 7.9407e-02, 1.6109e-02],
                 [-4.7459e-01, -1.5744e-01, -4.9599e-01, 1.7021e-01],
                 [-7.7234e-01, 5.2764e-01, 7.3166e-01, -7.0836e-02],
                 [9.7141e-01, -2.9385e-01, 1.4273e+00, -4.3123e-01],
                 [-9.7373e-01, -5.0459e-01, 6.2090e-01, 1.6899e+00],
                 [-1.8189e-01, -7.8022e-01, 3.1018e-02, -8.5824e-01]])

                module.weight = nn.Parameter(weight_tensor)
                return module.weight

        else:
            raise ValueError('Error: The size of the codebook vector is not 2x1 or 4x1')


#################################################################################################
#################################################################################################
##############################      Train Adaptive Codebook       ###############################
#################################################################################################
#################################################################################################


previous_active_vectors = None  # Start with empty codebook
criterion = nn.CrossEntropyLoss()

mobilenetv2_ac = AdapCB_Model(num_embeddings= 2,
                              codebook_size=NUM_EMBED,
                              quant=True, lambda_p=0.33)

# model.dropout = nn.Dropout(0.1, inplace=True)
# model.fc = nn.Linear(in_features=FEATURES, out_features=NUM_CLASSES, bias=True)

use_VQVAE = False
use_split = True
mixed_resolution = True


if use_split:
    mixed_resolution = [3 ,3 ,3, 3]
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(3033)
# #
# mobilenetv2_ac.quantizer.apply(init_weights)  # Initialize weights of codebook from normal distribution (0,1)
mobilenetv2_ac.quantizer.apply(lambda module: init_weights_lbg(module, cb_vec_dim=2, codebook_length=NUM_EMBED))
mobilenetv2_ac.to(device)


optimizer = torch.optim.Adam(mobilenetv2_ac.parameters(), lr=LEARNING_RATE) #weight_decay=1e-3

samples_for_scatter = []
vecs_to_save = []
# EPOCHS = [5, 5, 5, 5, 5, 4, 4, 4]   # Set number of epochs for each training phase.
EPOCHS = [1, 1, 15, 15, 15, 12 , 12, 15]  # Set number of epochs for each training phase.


if use_VQVAE == False: # Trained adaptive or multirate vector quantizer
    # range(int(np.log2(NUM_EMBED))):
    for level in  range(int(np.log2(NUM_EMBED))): # changed from range(1) - unquantized model

        print(f' \n ------------------------------------- ')
        print(f' \n Number of embeddings {NUM_EMBED}: \t Quantization level {level+1} ')
        print(f' \n ------------------------------------- ')

        num_active = pow(2, level + 1)  # Number of vectors-to-train in CB
        curr_vecs, encoder_samples = train(mobilenetv2_ac, optimizer, num_active, criterion, previous_active_vectors, EPOCHS[level], use_split=use_split, ADCs_resolutions=mixed_resolution)
        samples_for_scatter.append(encoder_samples)
        vecs_to_save.append(curr_vecs)
        previous_active_vectors = curr_vecs.clone().detach() # CHANGE when training
        scatter(vecs_to_save, samples_for_scatter)  # vec_to save are the codebook vectors m samples for scatter are random samples of the outbook of the encoder


else:
    num_active = NUM_EMBED
    curr_vecs, encoder_samples = train(mobilenetv2_ac, optimizer, num_active, criterion, previous_active_vectors,
                                       EPOCHS[0], use_split=False, ADCs_resolutions=mixed_resolution, use_VQVAE=use_VQVAE)
    samples_for_scatter.append(encoder_samples)
    vecs_to_save.append(curr_vecs)
    scatter(vecs_to_save,
            samples_for_scatter)



print(f' Number of codebook vectors { NUM_EMBED }')
logging.info(' Finished Training Adaptive Quantizer \n')


##########################################################################
############### Inference for Mixed Resolution ###########################
##########################################################################

#########################################################################################
""" Successive Refinement Code """

#
# class NestedAdaptiveVectorQuantizer(nn.Module):
#     def __init__(self, embedding_dim, codebook_size, commitment_cost=0.4, decay=0.8, epsilon=1e-5):
#         super(NestedAdaptiveVectorQuantizer, self).__init__()
#
#         self.d = embedding_dim  # The size of the vectors
#         self.p = codebook_size  # Number of vectors in the codebook
#
#         self.codebook = nn.Embedding(self.p, self.d)
#         self._commitment_cost = commitment_cost
#
#         self.register_buffer('_ema_cluster_size', torch.zeros(self.p))
#         # Buffers are tensors that are not updated during optimization
#         # but are used to keep track of some information within the model.
#
#         self._ema_w_arr = []  # These tensors are essentially weights used for the vector quantization
#         for level in range(int(np.log2(self.p))):
#             num_participants = 2 ** (level + 1)
#             self._ema_w = nn.Parameter(torch.Tensor(num_participants, self.d))
#             self._ema_w.data.normal_()  #initialized via a normal distribution
#             self._ema_w_arr.append(self._ema_w.to(device))
#
#         self._decay = decay
#         self._epsilon = epsilon
#
#         self.coefs = [0.56, 0.65, 0.69, 0.82, 0.9, 1, 1.2, 1.5]
#         self.coefs.reverse()
#
#     def quantizing(self, curr_cb, inputs, flat_input):
#
#         input_shape = inputs.shape
#
#         # Calculate distances
#         distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
#                      + torch.sum(curr_cb ** 2, dim=1)
#                      - 2 * torch.matmul(flat_input, curr_cb.t()))
#
#         # Encoding
#         encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
#         encodings = torch.zeros(encoding_indices.shape[0], curr_cb.shape[0], device=inputs.device)
#         encodings.scatter_(1, encoding_indices, 1)
#
#         # Quantize and unflatten
#         quantized = torch.matmul(encodings, curr_cb).view(input_shape)
#
#         return quantized, encodings
#
#     def number_to_pair(self, n): # Do not understand how this function relates to building the codebook
#         if n == 1:
#             return (0, 1)
#         else:
#             m = (n - 1) // 2
#             return (2 * m + 3 + (n % 2 == 0) - 1, 2 * m + 4 + (n % 2 == 0) - 1)
#
#     def build_codebook(self, prev_cb, stage):
#
#         indices = self.number_to_pair(stage)  # Get the current indices in the CB to add
#         vecs_to_add = self.codebook.weight[indices[0]: indices[1] + 1]  # Get the vectors in the CB to add
#
#         if stage == 1:
#             new_cb = vecs_to_add
#         else:
#             new_cb = prev_cb.unsqueeze(1) + vecs_to_add.unsqueeze(0)
#             new_cb = new_cb.view(-1, self.d)
#         return new_cb
#
#     def forward(self, inputs, num_vectors, prev_cbs):
#         # convert inputs from BCHW -> BHWC
#         inputs = inputs.permute(0, 2, 3, 1).contiguous()
#         input_shape = inputs.shape
#
#         # Flatten input
#         flat_input = inputs.reshape(-1, self.d)
#
#         quant_vecs = []
#         losses = []
#         cbs = []
#
#         if num_vectors == 2:
#             cbs.append(self.build_codebook(0, 1))
#         else:
#             new_cb = self.build_codebook(prev_cbs[-1], int(np.log2(num_vectors)))
#             for i in range(len(prev_cbs)):
#                 cbs.append(prev_cbs[i])
#             cbs.append(new_cb)
#
#         for level in range(int(np.log2(num_vectors))):
#             current_cb = cbs[level]
#             quantized, encodings = self.quantizing(current_cb, inputs, flat_input)
#             quant_vecs.append(quantized)
#
#             # if self.training:
#             #     # Loss
#             #     active_index = 2 ** (level + 1)
#
#             #     if level == 0:
#             #         # The ema_cluster_size counts how many time each vector on the codebook had been used
#             #         self._ema_cluster_size[:active_index] = self._ema_cluster_size[:active_index] * self._decay + \
#             #                                                 (1 - self._decay) * torch.sum(encodings, 0)[:active_index]
#
#             #         # total number of uses
#             #         n = torch.sum(self._ema_cluster_size.data)
#             #         self._ema_cluster_size[:active_index] = (
#             #                 (self._ema_cluster_size[:active_index] + self._epsilon)
#             #                 / (n + self.d * self._epsilon) * n
#             #         )
#             #         # The sum of the vectors in the inputs, according to their encoding
#             #         dw = torch.matmul(encodings.t(), flat_input)
#             #         self._ema_w_arr[level] = nn.Parameter(
#             #             self._ema_w_arr[level][:active_index] * self._decay + (1 - self._decay) * dw[
#             #                                                                                             :active_index])
#             #         ema_w = nn.Parameter(
#             #             self._ema_w_arr[level] / self._ema_cluster_size[:active_index].unsqueeze(1))
#             #         self.codebook.weight.data[:active_index] = ema_w
#
#             #     else:
#             #         # The ema_cluster_size counts how many time each vector on the codebook had been used
#             #         self._ema_cluster_size[active_index // 2:active_index] = self._ema_cluster_size[
#             #                                                                  active_index // 2:active_index] * self._decay + \
#             #                                                                  (1 - self._decay) * torch.sum(encodings,
#             #                                                                                                0)[
#             #                                                                                      active_index // 2:active_index]
#             #         # total number of uses
#             #         n = torch.sum(self._ema_cluster_size.data)
#             #         self._ema_cluster_size[active_index // 2:active_index] = (
#             #                 (self._ema_cluster_size[active_index // 2:active_index] + self._epsilon)
#             #                 / (n + self.d * self._epsilon) * n
#             #         )
#             #         # The sum of the vectors in the inputs, according to their encoding
#             #         dw = torch.matmul(encodings.t(), flat_input)
#             #         self._ema_w_arr[level] = nn.Parameter(
#             #             self._ema_w_arr[level] * self._decay + (1 - self._decay) * dw[:active_index])
#             #         ema_w = nn.Parameter(
#             #             self._ema_w_arr[level] / self._ema_cluster_size[:active_index].unsqueeze(1))
#             #         self.codebook.weight.data[active_index//2:active_index] = ema_w[active_index//2:active_index]
#
#             e_latent_loss = F.mse_loss(quantized.detach(), inputs)
#             q_latent_loss = F.mse_loss(inputs.detach(), quantized)
#             cb_loss = self.coefs[level]*q_latent_loss + self._commitment_cost * e_latent_loss  # + prox_loss  # codebook loss
#
#             quant_vecs[level] = inputs + (quant_vecs[level] - inputs).detach()  # gradient copying
#             quant_vecs[level] = quant_vecs[level].permute(0, 3, 1, 2).contiguous()
#
#             # else:
#             #     # convert quantized from BHWC -> BCHW
#             #     quant_vecs[level] = quant_vecs[level].permute(0, 3, 1, 2).contiguous()
#             #     cb_loss = 0
#
#             losses.append(cb_loss)
#
#             # print(f'Distortion: {F.mse_loss(flat_input, quant_vecs[level] .view(-1, 2))}')
#
#         return quant_vecs, losses, cbs
#
#     @property
#     def embedding(self):
#         return self._embedding
#
#
#
# ###################################################################################################
# ###################################################################################################
# #######################################       MODEL        ########################################
# ###################################################################################################
# ###################################################################################################
#
# class AdapCB_Model(nn.Module):
#     def __init__(self, num_embeddings=2, codebook_size=NUM_EMBED, lambda_c=0.1, lambda_p=0.33, quant=True):
#         super(AdapCB_Model, self).__init__()
#
#         self.encoder, self.decoder, self.classifier = self.split_net()
#         self.quantizer = NestedAdaptiveVectorQuantizer(num_embeddings, codebook_size, lambda_c)
#         self.quant = quant
#
#     def build_model(self, pretrained=True, fine_tune=True):
#         if pretrained:
#             print('[INFO]: Loading pre-trained weights')
#         elif not pretrained:
#             print('[INFO]: Not loading pre-trained weights')
#         inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 1], [6, 32, 3, 1], [6, 64, 4, 2], [6, 96, 3, 1],
#                                      [6, 160, 3, 1], [6, 320, 1, 1]]
#         model = models.mobilenet_v2(pretrained=pretrained, num_classes=1000, width_mult=1,
#                                     inverted_residual_setting=inverted_residual_setting)
#         if fine_tune:
#             print('[INFO]: Fine-tuning all layers...')
#             for params in model.parameters():
#                 params.requires_grad = True
#         elif not fine_tune:
#             print('[INFO]: Freezing hidden layers...')
#             for params in model.parameters():
#                 params.requires_grad = False
#
#         # change the final classification head, it is trainable,
#         # model.dropout = nn.Dropout(0.1,inplace=True)
#         model.fc = nn.Linear(in_features=FEATURES, out_features=NUM_CLASSES, bias=True)
#         return model
#
#     def split_net(self):
#         mobilenetv2 = self.build_model()
#
#         encoder = []
#         decoder = []
#         classifier = []
#
#         res_stop = 5
#         for layer_idx, l in enumerate(mobilenetv2.features):
#             if layer_idx <= res_stop:
#                 encoder.append(l)
#             else:
#                 decoder.append(l)
#
#         # classifier.append(mobilenetv2.dropout)
#         classifier.append(mobilenetv2.fc)
#
#         Encoder = nn.Sequential(*encoder)
#         Decoder = nn.Sequential(*decoder)
#         Classifier = nn.Sequential(*classifier)
#         return Encoder, Decoder, Classifier
#
#     def get_accuracy(self, gt, preds):
#         pred_vals = torch.max(preds.data, 1)[1]
#         batch_correct = (pred_vals == gt).sum()
#         return batch_correct
#
#     def normalize(self, inputs):
#         # Calculate the vector's magnitude
#         mean = inputs.mean()
#         output = inputs - mean
#         return output/(inputs.std())
#
#     def forward(self, inputs, num_active, prev_vecs):
#         z_e = self.encoder(inputs)
#         z_e = self.normalize(z_e)
#         if self.quant == True:
#             z_q, vq_loss, cbs = self.quantizer(z_e, num_active, prev_vecs)
#         else:
#             z_q, vq_loss, cbs = [z_e], [0], None
#
#         # print( f' Distortion between z_e and z_q using l2 norm {F.mse_loss(z_e.view(-1, self.quantizer.d).detach() , z_q[0].view(-1, self.quantizer.d).detach())}')
#
#
#         preds_list = []
#         for vecs in range(len(z_q)):
#             z_q_actives = z_q[vecs]
#             preds_list.append(self.decoder(z_q_actives))
#             preds_list[vecs] = preds_list[vecs].reshape(preds_list[vecs].shape[0],
#                                                         preds_list[vecs].shape[1] * preds_list[vecs].shape[2] *
#                                                         preds_list[vecs].shape[3])
#             preds_list[vecs] = self.classifier(preds_list[vecs])
#         return preds_list, vq_loss, cbs, z_e
#
#
# ###################################################################################################
# ###################################################################################################
# ##################################       Training Function        #################################
# ###################################################################################################
# ###################################################################################################
#
#
# def train(model, optimizer, num_active, criterion, prev_vecs=None, EPOCHS=20, commitment=0.05):
#     if num_active > 2:
#         checkpoint = torch.load('best_curr_model.pth')
#         model.load_state_dict(checkpoint)
#
#     start_time = time.time()
#     train_losses = []
#     acc_best = 0
#     encoder_samples = []
#     test_losses = []
#     val_losses = 0
#
#     subset_indices = torch.randperm(len(trainset))
#     subset = Subset(trainset, subset_indices)
#     subset_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
#
#     if num_active == 2:
#         loader = trainloader
#     else:
#         loader = subset_loader
#
#     for epc in range(EPOCHS):
#         train_acc = [0] * int(np.log2(num_active))
#         val_acc = [0] * int(np.log2(num_active))
#         trn_corr = [0] * int(np.log2(num_active))
#
#         model.train()
#         losses = 0
#         for batch_num, (Train, Labels) in enumerate(loader):
#             batch_num += 1
#             loss_levels = []
#             batch = Train.to(device)
#             preds_list, vq_loss, curr_cbs, z_e = model(batch, num_active, prev_vecs)
#
#             for q_level in range(len(preds_list)):
#                 ce_loss = criterion(preds_list[q_level], Labels.to(device))
#                 level_loss = ce_loss +  vq_loss[q_level]
#                 loss_levels.append(level_loss)
#                 train_acc[q_level] += model.get_accuracy(Labels.to(device), preds_list[q_level])
#
#             loss = sum(loss_levels) / len(loss_levels)
#             losses += loss.item()
#
#             # Update parameters
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#
#             if batch_num % 100 == 0:
#                 print(
#                     f'epoch: {epc + 1:2}  batch: {batch_num:2} [{BATCH_SIZE * batch_num:6}/{len(trainset)}]  total loss: {loss.item():10.8f}  \
#                 time = [{(time.time() - start_time) / 60}] minutes')
#
#         ### Accuracy ###
#         loss = losses / batch_num
#         train_losses.append(loss)
#         model.eval()
#         test_losses_val = 0
#
#         with torch.no_grad():
#
#             for b, (X_test, y_test) in enumerate(testloader):
#                 # Apply the model
#                 b += 1
#                 batch = X_test.to(device)
#                 val_preds, vq_val_loss, _, _ = model(batch, num_active, prev_vecs)
#                 loss_levels_val = []
#                 for q_level in range(len(val_preds)):
#                     ce_val_loss = criterion(val_preds[q_level], y_test.to(device))
#                     level_loss = ce_val_loss +  vq_val_loss[q_level]
#                     loss_levels_val.append(level_loss)
#                     val_acc[q_level] += model.get_accuracy(y_test.to(device), val_preds[q_level])
#
#                 val_loss = sum(loss_levels_val)
#                 val_losses += val_loss.item()
#
#             test_losses.append(val_losses / b)
#
#         total_train_acc = [100 * (acc.item()) / len(trainset) for acc in train_acc]
#         total_val_acc = [100 * (acc.item()) / len(testset) for acc in val_acc]
#
#         print(f'Train Models Accuracy at epoch {epc + 1} is {total_train_acc}%')
#         print(f'Validation Models Accuracy at epoch {epc + 1} is {total_val_acc}%')
#         if acc_best < total_val_acc[-1]:
#             acc_best = total_val_acc[-1]
#             best_model = model.state_dict()
#             torch.save(best_model, 'best_curr_model.pth')
#
#         if num_active == 2 and acc_best > 59:
#             break
#
#         model.train()
#
#     # curr_vecs = model.quantizer.codebook.weight.data[0:num_active].detach().clone()
#     encoder_samples = z_e
#     encoder_samples = encoder_samples.permute(0, 2, 3, 1).contiguous()
#     encoder_samples = encoder_samples.view(-1, 2)
#     duration = time.time() - start_time
#     print(f'Training took: {duration / 3600} hours')
#     return curr_cbs, encoder_samples
#
#
# ###################################################################################################
# ###################################################################################################
# ################################             Helpers                ###############################
# ###################################################################################################
# ###################################################################################################
#
#
# import matplotlib.pyplot as plt
#
# def scatter(cbs,train_points):
#     plt.rcParams["figure.figsize"] = (10,10)
#     colors = ['red', 'green', 'blue', 'purple', 'orange','magenta','cyan','yellow']
#     for i,level in enumerate(cbs):
#         x = np.array([elem[0] for elem in level.detach().cpu()])
#         y = np.array([elem[1] for elem in level.detach().cpu()])
#         name =  str(2*pow(2,i)) + 'Bits Vectors'
#         train_points_level = train_points[-1]
#         train_points_level = train_points_level[:20000]
#         train_x_vals = np.array([elem[0] for elem in train_points_level.detach().cpu()])
#         train_y_vals = np.array([elem[1] for elem in train_points_level.detach().cpu()])
#         plt.scatter(train_x_vals, train_y_vals,s=10, alpha=0.1,label = 'Train Vectors')
#         plt.scatter(x, y, s=250, alpha=1,label = name,c=colors[i % 8])
#
#         # Add axis labels and a title
#         plt.xlabel('X')
#         plt.ylabel('Y')
#         plt.title('2D Scatter Plot')
#         plt.grid()
#         plt.legend(loc='best')
#         # Show the plot
#         plt.show()
#
#
# # ###################################################################################################
# # ###################################################################################################
# # ################################      Train Adaptive Codebook       ###############################
# # ###################################################################################################
# # ###################################################################################################
#
# prev_vecs = None                        # start with empty previuos cb
# criterion = nn.CrossEntropyLoss()
# LEARNING_RATE = 0.00008
#
# adapcb_model = AdapCB_Model(num_embeddings=2, codebook_size=NUM_EMBED, lambda_c=0.05, lambda_p=0.33, quant=True)
#
# def init_weights(m):
#     if type(m) == nn.Embedding:
#         torch.nn.init.normal_(m.weight, mean=0, std=0.5)
#
# adapcb_model.quantizer.apply(init_weights)
# adapcb_model.to(device)
# optimizer = torch.optim.Adam(adapcb_model.parameters(), lr=LEARNING_RATE)
#
# samples_for_scatter = []
# vecs_to_save = []
#
# EPOCHS = [3,2,2,2,2,1,1,10]  # Set number of epochs for each training phase.
#
# for level in range(int(np.log2(NUM_EMBED))):
#     num_active = pow(2,level+1)        # Number of vectors-to-train in CB
#     curr_vecs, encoder_samples = train(adapcb_model,optimizer,num_active,criterion,prev_vecs,EPOCHS[level])
#     samples_for_scatter.append(encoder_samples)
#     vecs_to_save.append(curr_vecs[level])
#     scatter(vecs_to_save,samples_for_scatter)
#     prev_vecs = curr_vecs
#
# print(f' Number of codebook vectors {NUM_EMBED}')
# logging.info(' Finished Training Sucessive Refinement \n')

