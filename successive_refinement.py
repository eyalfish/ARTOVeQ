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
import pickle
import matplotlib.pyplot as plt
import tarfile
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder

SEED = 13

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



###################################################################################################
###################################################################################################
#################################   Globals & Hyperparameters     #################################
###################################################################################################
###################################################################################################

BATCH_SIZE = 64
LEARNING_RATE = 0.00001
ARCH = 'IMAGEWOOF'
NUM_EMBED = 256                                  # Number of vectors in the codebook.
SIZE = 64

###################################################################################################
###################################################################################################
############################################## Data ###############################################
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

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

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


class NestedAdaptiveVectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, codebook_size, commitment_cost=0.4, decay=0.8, epsilon=1e-5):
        super(NestedAdaptiveVectorQuantizer, self).__init__()

        self.d = embedding_dim  # The size of the vectors
        self.p = codebook_size  # Number of vectors in the codebook

        self.codebook = nn.Embedding(self.p, self.d)
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(self.p))
        # Buffers are tensors that are not updated during optimization
        # but are used to keep track of some information within the model.

        self._ema_w_arr = []  # These tensors are essentially weights used for the vector quantization
        for level in range(int(np.log2(self.p))):
            num_participants = 2 ** (level + 1)
            self._ema_w = nn.Parameter(torch.Tensor(num_participants, self.d))
            self._ema_w.data.normal_()  #initialized via a normal distribution
            self._ema_w_arr.append(self._ema_w.to(device))

        self._decay = decay
        self._epsilon = epsilon

        self.coefs = [0.56, 0.65, 0.69, 0.82, 0.9, 1, 1.2, 1.5]
        self.coefs.reverse()

    def quantizing(self, curr_cb, inputs, flat_input):

        input_shape = inputs.shape

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(curr_cb ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, curr_cb.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], curr_cb.shape[0], device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, curr_cb).view(input_shape)

        return quantized, encodings

    def number_to_pair(self, n): # Do not understand how this function relates to building the codebook
        if n == 1:
            return (0, 1)
        else:
            m = (n - 1) // 2
            return (2 * m + 3 + (n % 2 == 0) - 1, 2 * m + 4 + (n % 2 == 0) - 1)

    def build_codebook(self, prev_cb, stage):

        indices = self.number_to_pair(stage)  # Get the current indices in the CB to add
        vecs_to_add = self.codebook.weight[indices[0]: indices[1] + 1]  # Get the vectors in the CB to add

        if stage == 1:
            new_cb = vecs_to_add
        else:
            new_cb = prev_cb.unsqueeze(1) + vecs_to_add.unsqueeze(0)
            new_cb = new_cb.view(-1, self.d)
        return new_cb

    def forward(self, inputs, num_vectors, prev_cbs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.reshape(-1, self.d)

        quant_vecs = []
        losses = []
        cbs = []

        if num_vectors == 2:
            cbs.append(self.build_codebook(0, 1))
        else:
            new_cb = self.build_codebook(prev_cbs[-1], int(np.log2(num_vectors)))
            for i in range(len(prev_cbs)):
                cbs.append(prev_cbs[i])
            cbs.append(new_cb)

        for level in range(int(np.log2(num_vectors))):
            current_cb = cbs[level]
            quantized, encodings = self.quantizing(current_cb, inputs, flat_input)
            quant_vecs.append(quantized)

            # if self.training:
            #     # Loss
            #     active_index = 2 ** (level + 1)

            #     if level == 0:
            #         # The ema_cluster_size counts how many time each vector on the codebook had been used
            #         self._ema_cluster_size[:active_index] = self._ema_cluster_size[:active_index] * self._decay + \
            #                                                 (1 - self._decay) * torch.sum(encodings, 0)[:active_index]

            #         # total number of uses
            #         n = torch.sum(self._ema_cluster_size.data)
            #         self._ema_cluster_size[:active_index] = (
            #                 (self._ema_cluster_size[:active_index] + self._epsilon)
            #                 / (n + self.d * self._epsilon) * n
            #         )
            #         # The sum of the vectors in the inputs, according to their encoding
            #         dw = torch.matmul(encodings.t(), flat_input)
            #         self._ema_w_arr[level] = nn.Parameter(
            #             self._ema_w_arr[level][:active_index] * self._decay + (1 - self._decay) * dw[
            #                                                                                             :active_index])
            #         ema_w = nn.Parameter(
            #             self._ema_w_arr[level] / self._ema_cluster_size[:active_index].unsqueeze(1))
            #         self.codebook.weight.data[:active_index] = ema_w

            #     else:
            #         # The ema_cluster_size counts how many time each vector on the codebook had been used
            #         self._ema_cluster_size[active_index // 2:active_index] = self._ema_cluster_size[
            #                                                                  active_index // 2:active_index] * self._decay + \
            #                                                                  (1 - self._decay) * torch.sum(encodings,
            #                                                                                                0)[
            #                                                                                      active_index // 2:active_index]
            #         # total number of uses
            #         n = torch.sum(self._ema_cluster_size.data)
            #         self._ema_cluster_size[active_index // 2:active_index] = (
            #                 (self._ema_cluster_size[active_index // 2:active_index] + self._epsilon)
            #                 / (n + self.d * self._epsilon) * n
            #         )
            #         # The sum of the vectors in the inputs, according to their encoding
            #         dw = torch.matmul(encodings.t(), flat_input)
            #         self._ema_w_arr[level] = nn.Parameter(
            #             self._ema_w_arr[level] * self._decay + (1 - self._decay) * dw[:active_index])
            #         ema_w = nn.Parameter(
            #             self._ema_w_arr[level] / self._ema_cluster_size[:active_index].unsqueeze(1))
            #         self.codebook.weight.data[active_index//2:active_index] = ema_w[active_index//2:active_index]

            e_latent_loss = F.mse_loss(quantized.detach(), inputs)
            q_latent_loss = F.mse_loss(inputs.detach(), quantized)
            cb_loss = self.coefs[level]*q_latent_loss + self._commitment_cost * e_latent_loss  # + prox_loss  # codebook loss

            quant_vecs[level] = inputs + (quant_vecs[level] - inputs).detach()  # gradient copying
            quant_vecs[level] = quant_vecs[level].permute(0, 3, 1, 2).contiguous()

            # else:
            #     # convert quantized from BHWC -> BCHW
            #     quant_vecs[level] = quant_vecs[level].permute(0, 3, 1, 2).contiguous()
            #     cb_loss = 0

            losses.append(cb_loss)

        return quant_vecs, losses, cbs

    @property
    def embedding(self):
        return self._embedding



###################################################################################################
###################################################################################################
#######################################       MODEL        ########################################
###################################################################################################
###################################################################################################

class AdapCB_Model(nn.Module):
    def __init__(self, num_embeddings=2, codebook_size=NUM_EMBED, lambda_c=0.1, lambda_p=0.33, quant=True):
        super(AdapCB_Model, self).__init__()

        self.encoder, self.decoder, self.classifier = self.split_net()
        self.quantizer = NestedAdaptiveVectorQuantizer(num_embeddings, codebook_size, lambda_c)
        self.quant = quant

    def build_model(self, pretrained=True, fine_tune=True):
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
        # model.dropout = nn.Dropout(0.2,inplace=True)
        model.fc = nn.Linear(in_features=FEATURES, out_features=NUM_CLASSES, bias=True)
        return model

    def split_net(self):
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

        # classifier.append(mobilenetv2.dropout)
        classifier.append(mobilenetv2.fc)

        Encoder = nn.Sequential(*encoder)
        Decoder = nn.Sequential(*decoder)
        Classifier = nn.Sequential(*classifier)
        return Encoder, Decoder, Classifier

    def get_accuracy(self, gt, preds):
        pred_vals = torch.max(preds.data, 1)[1]
        batch_correct = (pred_vals == gt).sum()
        return batch_correct

    def normalize(self, inputs):
        # Calculate the vector's magnitude
        mean = inputs.mean()
        output = inputs - mean
        return output/(inputs.std())

    def forward(self, inputs, num_active, prev_vecs):
        z_e = self.encoder(inputs)
        z_e = self.normalize(z_e)
        if self.quant == True:
            z_q, vq_loss, cbs = self.quantizer(z_e, num_active, prev_vecs)
        else:
            z_q, vq_loss, cbs = [z_e], [0], None

        preds_list = []
        for vecs in range(len(z_q)):
            z_q_actives = z_q[vecs]
            preds_list.append(self.decoder(z_q_actives))
            preds_list[vecs] = preds_list[vecs].reshape(preds_list[vecs].shape[0],
                                                        preds_list[vecs].shape[1] * preds_list[vecs].shape[2] *
                                                        preds_list[vecs].shape[3])
            preds_list[vecs] = self.classifier(preds_list[vecs])
        return preds_list, vq_loss, cbs, z_e


###################################################################################################
###################################################################################################
##################################       Training Function        #################################
###################################################################################################
###################################################################################################
from torch.utils.data import DataLoader, Subset

def train(model, optimizer, num_active, criterion, prev_vecs=None, EPOCHS=20, commitment=0.2):
    if num_active > 2:
        checkpoint = torch.load('best_curr_model.pth')
        model.load_state_dict(checkpoint)

    start_time = time.time()
    train_losses = []
    acc_best = 0
    encoder_samples = []
    test_losses = []
    val_losses = 0

    subset_indices = torch.randperm(len(trainset))
    subset = Subset(trainset, subset_indices)
    subset_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)

    if num_active == 2:
        loader = trainloader
    else:
        loader = subset_loader

    for epc in range(EPOCHS):
        train_acc = [0] * int(np.log2(num_active))
        val_acc = [0] * int(np.log2(num_active))
        trn_corr = [0] * int(np.log2(num_active))

        model.train()
        losses = 0
        for batch_num, (Train, Labels) in enumerate(loader):
            batch_num += 1
            loss_levels = []
            batch = Train.to(device)
            preds_list, vq_loss, curr_cbs, z_e = model(batch, num_active, prev_vecs)

            for q_level in range(len(preds_list)):
                ce_loss = criterion(preds_list[q_level], Labels.to(device))
                level_loss = ce_loss + commitment * vq_loss[q_level]
                loss_levels.append(level_loss)
                train_acc[q_level] += model.get_accuracy(Labels.to(device), preds_list[q_level])

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

        with torch.no_grad():

            for b, (X_test, y_test) in enumerate(testloader):
                # Apply the model
                b += 1
                batch = X_test.to(device)
                val_preds, vq_val_loss, _, _ = model(batch, num_active, prev_vecs)
                loss_levels_val = []
                for q_level in range(len(val_preds)):
                    ce_val_loss = criterion(val_preds[q_level], y_test.to(device))
                    level_loss = ce_val_loss + commitment * vq_val_loss[q_level]
                    loss_levels_val.append(level_loss)
                    val_acc[q_level] += model.get_accuracy(y_test.to(device), val_preds[q_level])

                val_loss = sum(loss_levels_val)
                val_losses += val_loss.item()

            test_losses.append(val_losses / b)

        total_train_acc = [100 * (acc.item()) / len(trainset) for acc in train_acc]
        total_val_acc = [100 * (acc.item()) / len(testset) for acc in val_acc]

        print(f'Train Models Accuracy at epoch {epc + 1} is {total_train_acc}%')
        print(f'Validation Models Accuracy at epoch {epc + 1} is {total_val_acc}%')
        if acc_best < total_val_acc[-1]:
            acc_best = total_val_acc[-1]
            best_model = model.state_dict()
            torch.save(best_model, 'best_curr_model.pth')

        if num_active == 2 and acc_best > 59:
            break

        model.train()

    # curr_vecs = model.quantizer.codebook.weight.data[0:num_active].detach().clone()
    encoder_samples = z_e
    encoder_samples = encoder_samples.permute(0, 2, 3, 1).contiguous()
    encoder_samples = encoder_samples.view(-1, 2)
    duration = time.time() - start_time
    print(f'Training took: {duration / 3600} hours')
    return curr_cbs, encoder_samples


###################################################################################################
###################################################################################################
################################             Helpers                ###############################
###################################################################################################
###################################################################################################


import matplotlib.pyplot as plt

def scatter(cbs,train_points):
    plt.rcParams["figure.figsize"] = (10,10)
    colors = ['red', 'green', 'blue', 'purple', 'orange','magenta','cyan','yellow']
    for i,level in enumerate(cbs):
        x = np.array([elem[0] for elem in level.detach().cpu()])
        y = np.array([elem[1] for elem in level.detach().cpu()])
        name =  str(2*pow(2,i)) + 'Bits Vectors'
        train_points_level = train_points[-1]
        train_points_level = train_points_level[:20000]
        train_x_vals = np.array([elem[0] for elem in train_points_level.detach().cpu()])
        train_y_vals = np.array([elem[1] for elem in train_points_level.detach().cpu()])
        plt.scatter(train_x_vals, train_y_vals,s=10, alpha=0.1,label = 'Train Vectors')
        plt.scatter(x, y, s=250, alpha=1,label = name,c=colors[i % 8])

        # Add axis labels and a title
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('2D Scatter Plot')
        plt.grid()
        plt.legend(loc='best')
        # Show the plot
        plt.show()


# ###################################################################################################
# ###################################################################################################
# ################################      Train Adaptive Codebook       ###############################
# ###################################################################################################
# ###################################################################################################

prev_vecs = None                        # start with empty previuos cb
criterion = nn.CrossEntropyLoss()
LEARNING_RATE = 0.00001

adapcb_model = AdapCB_Model(num_embeddings=2, codebook_size=NUM_EMBED, lambda_c=0.1, lambda_p=0.33, quant=True)

def init_weights(m):
    if type(m) == nn.Embedding:
        torch.nn.init.normal_(m.weight, mean=0, std=0.5)

adapcb_model.quantizer.apply(init_weights)
adapcb_model.to(device)
optimizer = torch.optim.Adam(adapcb_model.parameters(), lr=LEARNING_RATE)

samples_for_scatter = []
vecs_to_save = []

EPOCHS = [1, 1, 1, 1, 1, 1, 1,5]  # Set number of epochs for each training phase.

for level in range(int(np.log2(NUM_EMBED))):
    num_active = pow(2,level+1)        # Number of vectors-to-train in CB
    curr_vecs, encoder_samples = train(adapcb_model, optimizer, num_active, criterion, prev_vecs,EPOCHS[level])
    samples_for_scatter.append(encoder_samples)
    vecs_to_save.append(curr_vecs[level])
    scatter(vecs_to_save,samples_for_scatter)
    prev_vecs = curr_vecs

