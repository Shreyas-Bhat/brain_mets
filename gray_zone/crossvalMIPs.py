from __future__ import print_function

import os
import shutil
import tempfile
import time
import sys
import csv
from glob import glob

import random
import numpy as np
from numpy.random import shuffle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold

from monai.config import print_config
from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    AddChannel,
    AsChannelFirst,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    ScaleIntensityRange,
    Transpose,
    Resize,
    ToTensor,
    EnsureType,
)
from monai.utils import set_determinism

# print_config()

base_dir = '/wkdir2/MIP_data/'
#folders = ['train_nii/']
folders = ['total_set/']#, 'test_nii/'] # ['train_nii/', 'test_nii/']
inputs_to_process = ['cube.npy','cube_uint8.npy']

for folder in folders:
    patient_dir = base_dir + folder
    patients = [os.path.relpath(directory_paths, patient_dir) for (directory_paths, directory_names, filenames) in os.walk(patient_dir) if all(input_to_process in filenames for input_to_process in inputs_to_process)]
    patients.sort()
    #if folder == 'train_nii/':
    if folder == 'total_set/':
        patients_train = patients
    # if folder == 'tot':
    #     patients_test = patients
    
all_data = pd.read_csv('/wkdir2/MVtmb_new (2)_deidentified.csv')
all_image = list(all_data["Martinos ID"])
all_image = [str(i) for i in all_image]
all_image = [i for i in all_image if i.startswith('MEL') or i.startswith('LUN') or i.startswith('BRE')]
import os

directory = '/wkdir2/MIP_data/total_set/'
all_image = [i for i in all_image if i.startswith('MEL') or i.startswith('LUN') or i.startswith('BRE')]
# print(all_image)

# create an empty list to store the files
files_in_directory = []

for i in all_image:
    if os.path.exists(os.path.join(directory, i)):
        files_in_directory.append(i)
        
# print(f'Files present in {directory} are: {files_in_directory}')
import numpy as np

# condition = [    (all_data['tmb'] <= 5),
#     (all_data['tmb'] <= 22),
#     (all_data['tmb'] > 22)
# ]
condition = [
    (all_data['tmb'] <= 12),
    (all_data['tmb'] > 12)
]

choices = [0, 1]

all_data['count'] = np.select(condition, choices)


for i in range(len(all_image)):
    b = all_image[i]
    b_str = str(b)
    c = b_str.zfill(5)
    all_image[i] = c
    
all_data["paired"].fillna(0, inplace=True)
all_label = list(all_data["count"])
print("all_label", len(all_label))
missing_index = []
missing_image = []
missing_label = []

for i in all_image:
    if i not in patients_train:
        i_ind = all_image.index(i)
        missing_index.append(i_ind)
        missing_image.append(all_image[i_ind]) 
        missing_label.append(all_label[i_ind]) 

for i in sorted(missing_index, reverse=True):
    del all_image[i]
    del all_label[i]

for a, b in zip(missing_image, missing_label):
    print(a,b)
    
if all(i in patients_train for i in all_image):
    print('The lists are identical')
else:
    print('Error: the lists are not identical')

# for i in range(len(patients_train)):
#     if all_image[i] == patients_train[i]:
#         pass
#     else:
#         print(f'The two lists do not match at position {i}')

# Using train_test_split of aklearn to get internal train and internal test sets. The internal train set will be split 4:1
# for each fold in the k-fold cross validation. This initial split is strictly done to evaluate model performance on a predefined
# test set that has known results. The external test set == ktest is the test set provided with UNKNOWN results
# print(type(folder))
# set_determinism(seed=0)
# torch.manual_seed(0)
# random.seed(0)
# np.random.seed(0)

# need to know if any of the above/subset is necessary
all_image_path = [base_dir + folder + i + '/cube_uint8.npy' for i in files_in_directory]

ktrain_image_path = np.array(all_image_path)
ktrain_label = np.array(all_label)
print(ktrain_image_path.shape, len(all_label))

# pathFileiTrain, pathFileiTest, labeliTrain, labeliTest = train_test_split(ktrain_image_path, ktrain_label, test_size=0.1, random_state=0, stratify=ktrain_label)

# print(f'\nInternal train dataset length = {len(pathFileiTrain)}')
# print(f'Count of 0s in internal train dataset = {np.count_nonzero(labeliTrain==0)}')
# print(f'Count of 1s in internal train dataset = {np.count_nonzero(labeliTrain)}')
# #
# print(f'Internal test dataset length, fold = {len(pathFileiTest)}')
# print(f'Count of 0s in internal test dataset, fold = {np.count_nonzero(labeliTest==0)}')
# print(f'Count of 1s in internal test dataset, fold = {np.count_nonzero(labeliTest)}')
# #
#trBatchSize = 32
#
# all_image_path = [base_dir + folder + i + '/cube_uint8.npy' for i in files_in_directory]

val_frac = 0.15
test_frac = 0.15
length = len(all_image_path)
indices = np.arange(length)
np.random.shuffle(indices)

test_split = int(test_frac * length)
val_split = int(val_frac * length) + test_split
test_indices = indices[:test_split]
val_indices = indices[test_split:val_split]
train_indices = indices[val_split:]

pathFileTrain = [all_image_path[i] for i in train_indices]
labelTrain = [int(all_label[i]) for i in train_indices]
pathFileValid = [all_image_path[i] for i in val_indices]
labelValid = [int(all_label[i]) for i in val_indices]
pathFileiTest = [all_image_path[i] for i in test_indices]
labeliTest = [int(all_label[i]) for i in test_indices]

print(
    f"Training count: {len(pathFileTrain)}, Validation count: "
    f"{len(pathFileValid)}, internal Test count: {len(pathFileiTest)}")

# Create list of transformations using Pytorch transforms
num_class=2
# Training
transformList = []
#transformList.append(transforms.RandomHorizontalFlip(p=0.5)) 
transformList.append(transforms.RandomVerticalFlip(p=0.5))
transformList.append(transforms.RandomAffine(20, translate=(0.05,0.05), scale=(0.75,1.35)))
# transformList.append(transforms.RandomRotation(20))
# transformList.append(transforms.functional.adjust_gamma())
transformList.append(transforms.ToTensor())
transformSequence_train = transforms.Compose(transformList)

# Additional options to consider
#transformList.append(transforms.functional.adjust_gamma())
#normalize = transforms.Normalize([0.485*255, 0.456*255, 0.406*255], [0.229*255, 0.224*255, 0.225*255])
#transformList.append(normalize) # for both train and validation sequences

# Validation
transformList = []
transformList.append(transforms.ToTensor())
transformSequence_valid = transforms.Compose(transformList)

y_pred_trans = Compose([EnsureType(), Activations(softmax=True)])   
# changed softmax to sigmoid initially (thinking binary classification - however I am treating binary as a multiclass problem, 0 and 1. 
# then removed softmax/sigmoid because CrossEntropyLoss() already applies a LogSoftmax combined with NLLLoss
# however, where I use the softmax function: i.e. in the validation step, it is likely necessary for computing AUC, so added it back per monai example
# applied during model run at the end of each batch for each epoch: y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)] where y_pred = torch.cat([y_pred, model(val_images)], dim=0)
# softmax function applied as a last layer over the predicted outputs for the validation set, on each patient (i.e. by patient)
# The softmax function takes as input a vector z of K real numbers, and normalizes it into a probability distribution consisting of K probabilities proportional to the exponentials of the input numbers. 
# That is, prior to applying softmax, some vector components could be negative, or greater than one; and might not sum to 1; but after applying softmax, each component will be in the interval [0,1]  
# and the components will add up to 1, so that they can be interpreted as probabilities. Furthermore, the larger input components will correspond to larger probabilities. 
y_trans = Compose([EnsureType(), AsDiscrete(to_onehot=2, num_classes=2)])
# applied during model run at the end of each batch for each epoch: y_onehot = [y_trans(i) for i in decollate_batch(y)] where y = torch.cat([y, val_labels], dim=0)
# AsDiscrete function essentially converts the labels of the validation set, on each patient (i.e. by patient), into discrete values and to the onehot format 
# A one hot encoding is a representation of categorical variables as binary vectors. This first requires that the categorical values be mapped to integer values in this case 0 and 1.
# Then, each integer value is represented as a binary vector that is all zero values except the index of the integer, which is marked with a 1.
# As an example here, 0 corresponds to the binary vector [1,0] with the zeroth index marked as 1 (True)
# 1 corresponds to the binary vector [0,1] with the first (1th) index marked as 1 (True)
y_test_trans = Compose([EnsureType()])#, AsDiscrete(to_onehot=True, num_classes=num_class)])


# Dataset class and collate function: stacking after transforms is done on an orientation (ax, co, sag) basis
# Each .npy stack has 3 views of 4 modalities (FLAIR x-axial, y-coronal, z-sagital, T2, T1post, T1pre) - 12 stack in order; 
# .npy stack generation script in compile_cubes_25D_rakin.py and MGMT_classification_preprocessing.ipynb

class kaggle_MGMT(Dataset):
    def __init__(self, image_list, label_list, transform=None):
        self.image_names = image_list
        self.labels = label_list
        self.transform = transform
    #
    def __getitem__(self, index):
        # try:
        cube_uint8 = np.load(self.image_names[index])
        # cube_uint8_4_ax = np.stack((cube_uint8[:,:,0],cube_uint8[:,:,3],cube_uint8[:,:,6]),axis=2)
        cube_uint8_4_ax = np.stack(cube_uint8[:,:,8])
        # cube_uint8_4_sa = np.stack((cube_uint8[:,:,2],cube_uint8[:,:,5],cube_uint8[:,:,8]),axis=2)
        # cube_uint8_4_ax = np.stack((cube_uint8[:,:,0],cube_uint8[:,:,3],cube_uint8[:,:,6]),axis=2)
        # cube_uint8_4_ax = np.stack((cube_uint8[:,:,0],cube_uint8[:,:,2],cube_uint8[:,:,4]),axis=2)
        # cube_uint8_4_co = np.stack((cube_uint8[:,:,1],cube_uint8[:,:,3],cube_uint8[:,:,5]),axis=2)
        # cube_uint8_4_sa = np.stack((cube_uint8[:,:,2],cube_uint8[:,:,4],cube_uint8[:,:,6]),axis=2)

#         print(cube_uint8_4_ax.shape)
#         print(cube_uint8_4_co.shape)
#         print(cube_uint8_4_sa.shape)
        
        im_4_ax = Image.fromarray(np.resize(cube_uint8_4_ax, (200,200,3)),'RGB')
        

#         im_4_co = Image.fromarray(np.resize(cube_uint8_4_co, (200,200,3)),'RGB')
        

#         im_4_sa = Image.fromarray(np.resize(cube_uint8_4_sa, (200,200,3)),'RGB')
        


        # print(im_4_ax.size)
        # print(im_4_co.size)
        # print(im_4_sa.size)
        # im_4_ax = Image.fromarray(cube_uint8_4_ax,'RGB')
        # im_4_co = Image.fromarray(cube_uint8_4_co,'RGB')
        # im_4_sa = Image.fromarray(cube_uint8_4_sa,'RGB')

        if self.transform is not None:
            #print("here")
            # im_4_ax = torch.from_numpy(np.array(im_4_ax))
            # im_4_co = torch.from_numpy(np.array(im_4_co))
            # im_4_sa = torch.from_numpy(np.array(im_4_sa))
            # image = torch.cat([im_4_ax, im_4_co, im_4_sa])
            # im_4_ax = torch.unsqueeze(im_4_ax, dim=0)
            # im_4_co = torch.unsqueeze(im_4_co, dim=0)
            # im_4_sa = torch.unsqueeze(im_4_sa, dim=0)
            im_4_ax_tup = self.transform(im_4_ax)
            # im_4_co_tup = self.transform(im_4_co)
            # im_4_sa_tup = self.transform(im_4_sa)
#             print("im_ax", im_4_ax_tup.shape)
#             print("im_co", im_4_co_tup.shape)
#             print("im_sa", im_4_sa_tup.shape)
            
            # image = torch.cat(([im_4_ax_tup[0], im_4_co_tup[0], im_4_sa_tup[0]]), dim=0)
#             image = torch.cat((im_4_ax_tup[0,:,:].unsqueeze_(0), im_4_co_tup[0,:,:].unsqueeze_(0), im_4_sa_tup[0,:,:].unsqueeze_(0),
#                                im_4_ax_tup[1,:,:].unsqueeze_(0), im_4_co_tup[1,:,:].unsqueeze_(0), im_4_sa_tup[1,:,:].unsqueeze_(0),
#                                im_4_ax_tup[2,:,:].unsqueeze_(0), im_4_co_tup[2,:,:].unsqueeze_(0), im_4_sa_tup[2,:,:].unsqueeze_(0),),0)
                                
            

            #image = self.transform(image)
            # print("im_4_ax", im_4_ax)
            # print("im_4_co", im_4_co)
            # print("im_4_sa", im_4_sa)
            # convert PIL images to tensors
            # im_4_ax = torch.from_numpy(np.array(im_4_ax))
            # im_4_co = torch.from_numpy(np.array(im_4_co))
            # im_4_sa = torch.from_numpy(np.array(im_4_sa))

        # if im_4_ax is None or im_4_co is None or im_4_sa is None:
        if im_4_ax is None:
            #print("hello")
            return None

        #print("moshi")
        # image = torch.cat(torch.stack((im_4_ax, im_4_co, im_4_sa)),0) # stack by orientation, replace alternative here
        #image = torch.stack([im_4_ax, im_4_co, im_4_sa])
        #for i in (im_4_ax, im_4_co, im_4_sa):
        #   print(type(i))
        # image = torch.cat(([im_4_ax_tup, im_4_co_tup, im_4_sa_tup]), dim=0)
        image = torch.cat(([im_4_ax_tup]), dim=0)

        #print("concat",image.shape)
        label = torch.tensor(self.labels[index])
        return image, label
           # print("how")
        # except:
        #     print("hello1")
        #     return None
    #
    def __len__(self):
        return len(self.image_names)

#
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch)) 
    # list(filter batch through lambda function where x is not None)
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# Sample each class equally during training
def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val]
    return weight

class ResNet18(nn.Module):
    def __init__(self, in_channels=3):
        super(ResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        self.resnet18.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Sequential(
            nn.Linear(num_ftrs, 2))
    #
    def forward(self, x):
        x = self.resnet18(x)
        return x
    
# Name and directory to save model in
modelname =  'resnet18_cv_33'
datadir = '/wkdir2/code/BrainMets/models/'
# Path for model checkpoints
checkpointDir = '/wkdir2/code/BrainMets/checkpoints/'
# Path of file for listing losses for each iteration
lossFilePath = '/wkdir2/code/BrainMets/'+modelname+'_loss'

# StratifiedKFold  on internal train set: pathFileiTrain and labeliTrain

num_folds = 5
start = time.time()

skf = StratifiedKFold(n_splits=num_folds)

fold_loss_train_values = []
fold_loss_val_values = []
fold_metric_values = []
fold_accuracy_values = []

for fold, (train_index, validation_index) in enumerate(skf.split(pathFileTrain, labelTrain)):
    # Define path and label lists, and print relevant values for each fold
    print(f'\nFOLD: {fold+1}')
    print("-" * 10)
    # pathFileTrain, pathFileValid = pathFileiTrain[train_index], pathFileiTrain[validation_index]
    # labelTrain, labelValid = labeliTrain[train_index], labeliTrain[validation_index]
    #
    print(f'\ntrain path list length, fold {fold+1} = {len(pathFileTrain)}')
    print(f'Count of 0s in train dataset, fold {fold+1} = {np.count_nonzero(labelTrain==0)}')
    print(f'Count of 1s in train dataset, fold {fold+1} = {np.count_nonzero(labelTrain)}')
    #
    print(f'validation path list length, fold {fold+1} = {len(pathFileValid)}')
    print(f'Count of 0s in validation dataset, fold {fold+1} = {np.count_nonzero(labelValid==0)}')
    print(f'Count of 1s in validation dataset, fold {fold+1} = {np.count_nonzero(labelValid)}')
    #
    # Assign weights to ensure balanced classes when sampling the kth-fold batches for train set
    weights  = make_weights_for_balanced_classes(labelTrain,2)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    #
    trBatchSize = 32
    #
    datasetTrain = kaggle_MGMT(pathFileTrain, labelTrain, transformSequence_train)
    datasetValid = kaggle_MGMT(pathFileValid, labelValid, transformSequence_valid)
    #
    print(f'\ntrain dataset length, fold {fold+1} = {len(datasetTrain)}')
    print(f'validation dataset length, fold {fold+1} = {len(datasetValid)}')
    #
    dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=False, sampler=sampler, num_workers=24, pin_memory=True, collate_fn=collate_fn)
    dataLoaderValid = DataLoader(dataset=datasetValid, batch_size=trBatchSize, shuffle=False, num_workers=24, pin_memory=True, collate_fn=collate_fn)
        
    ### Modelling part
    print(f'\nMODEL: {fold+1}')
    print("-" * 10)
    
    ## initiate device and neural net for this fold
    deviceNum = 0
    device = torch.device(deviceNum if torch.cuda.is_available() else "cpu")
    model = ResNet18().to(device)
    loss_function = nn.CrossEntropyLoss() # nn.BCEWithLogitsLoss()
    # checkpoint = torch.load('/wkdir2/code/BrainMets/models/resnet18_co_3.pth.tar')
    # model.load_state_dict(checkpoint)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
    optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    trMaxEpoch = 200
    val_interval = 1
    auc_metric = ROCAUCMetric()
    
    ## define modelname, directory_path
    modelname_fold = modelname + '_' + str(fold+1)
    
    ## define metrics true for all epochs in this fold
    bestvalloss = 100
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_train_values = []
    epoch_loss_val_values = []
    metric_values = []
    accuracy_values = []
    
    ## run the model for this fold
    for epoch in range(trMaxEpoch):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{trMaxEpoch}")
        model.train()
        epoch_loss_train = 0
        step = 0
        for batch_data in dataLoaderTrain:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)        
            #labels = (labels.unsqueeze(1)).float() # changed this for BCEwithLogits loss (removed sigmoid)
            optimizer.zero_grad()
            outputs = model(inputs)
            trainloss = loss_function(outputs, labels)
            trainloss.backward()
            optimizer.step()
            epoch_loss_train += trainloss.item()
            print(
                f"{step}/{len(datasetTrain) // dataLoaderTrain.batch_size}, "
                f"train_loss: {trainloss.item():.4f}")
            epoch_len = len(datasetTrain) // dataLoaderTrain.batch_size
        epoch_loss_train /= step  
        epoch_loss_train_values.append(epoch_loss_train)
        print(f"epoch {epoch + 1} average train loss: {epoch_loss_train:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            epoch_loss_val = 0
            step = 0
            #
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in dataLoaderValid:
                    step += 1
                    val_images, val_labels = (
                        val_data[0].to(device),
                        val_data[1].to(device),
                    )                
                    #val_labels = (val_labels.unsqueeze(1)).float() # changed this for BCEwithLogits loss (removed sigmoid)
                    val_outputs = model(val_images) #added
                    validloss = loss_function(val_outputs, val_labels) #added
                    scheduler.step(validloss)
                    y_pred = torch.cat([y_pred, val_outputs], dim=0) 
                    y = torch.cat([y, val_labels], dim=0)
                    #validloss = loss_function(y_pred, y) #added
                    epoch_loss_val += validloss.item()
                #
                epoch_loss_val /= step
                epoch_loss_val_values.append(epoch_loss_val)
                print(f"epoch {epoch + 1} average validation loss: {epoch_loss_val:.4f}")
                #
                y_onehot = [y_trans(i) for i in decollate_batch(y)]
                y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
                auc_metric(y_pred_act, y_onehot)
                result = auc_metric.aggregate()
                auc_metric.reset()
                del y_pred_act, y_onehot
                metric_values.append(result)
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                # argmax returns index of max value in the first dimension of y_pred (a tensor of probabilities). The index is either 0 or 1 for binary classification
                # y is a tensor with either 0 or 1 as the label value. torch.eq computes the element by element equality and returns a boolean tensor
                # .sum() then adds all the boolean true's (where input and predictions match, and .item() converts this to a python number)
                # acc_metric then computes the accuracy as correct/total
                acc_metric = acc_value.sum().item() / len(acc_value)
                accuracy_values.append(acc_metric) # at the epoch level
                #
                if result > best_metric:
                    best_metric = result
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), datadir+modelname_fold+'.pth.tar')
                    print("saved new best metric model")
                #
                if epoch_loss_val < bestvalloss:
                    bestvalloss = epoch_loss_val
                    bestvalloss_epoch = epoch + 1
                #
                print(
                    f"current epoch: {epoch + 1} current AUC: {result:.4f}"
                    f" current accuracy: {acc_metric:.4f}"
                    f" best AUC: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                    f" validation loss: {epoch_loss_val:.4f}"
                    f" at epoch: {epoch}"
                )

    #
    print(
        f"fold {fold+1} train completed, best metric: {best_metric:.4f}"
        f" at epoch: {best_metric_epoch}"
        f" validation loss: {epoch_loss_val:.4f}"
        f" at epoch: {epoch}")
    

    #
    fold_loss_train_values.append(epoch_loss_train_values)
    fold_loss_val_values.append(epoch_loss_val_values)
    fold_metric_values.append(metric_values)
    fold_accuracy_values.append(accuracy_values)
    avg_train_loss = np.mean(fold_loss_train_values,axis=0)
    avg_val_loss = np.mean(fold_loss_val_values,axis=0)
    avg_metric = np.mean(fold_metric_values,axis=0)
    avg_accuracy = np.mean(fold_accuracy_values,axis=0)
    plt.figure("figures", (20, 6))

    plt.subplot(1, 3, 1)
    labels = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
    colors = ['red', 'green', 'blue', 'orange', 'brown']
    plt.title("Epoch Average Loss: Train and Validation")
    x1 = [i + 1 for i in range(len(avg_train_loss))]
    x2 = [val_interval * (i + 1) for i in range(len(avg_val_loss))]
    y1 = avg_train_loss
    y2 = avg_val_loss
    plt.xlabel("epoch")
    ax1 = plt.plot(x1, y1, '-b', x2, y2, '-g')
    plt.ylim([0,1])
    plt.subplot(1, 3, 2)
    plt.title("Epoch Average Val AUC")
    x = [val_interval * (i + 1) for i in range(len(avg_metric))]
    y = avg_metric
    plt.xlabel("epoch")
    ax3 = plt.plot(x, y, label=labels[fold], color=colors[fold])

    plt.subplot(1, 3, 3)
    plt.title("Epoch Average Val Accuracy")
    x = [val_interval * (i + 1) for i in range(len(avg_accuracy))]
    y = avg_accuracy

    
    plt.xlabel("epoch")
    ax3 = plt.plot(x, y, label=labels[fold], color=colors[fold])

    plt.savefig(f"/wkdir2/code/BrainMets/plots/resnet18_cv_33/plots{fold}.png")
    plt.show()

    print(f"\n train loss {avg_train_loss[len(avg_train_loss)-1]} at epoch {len(avg_train_loss)}")
    print(f" val loss {avg_val_loss[len(avg_val_loss)-1]} at epoch {len(avg_val_loss)}")
    print(f" val AUC {avg_metric[len(avg_metric)-1]} at epoch {len(avg_metric)}")
    print(f" val accuracy {avg_accuracy[len(avg_accuracy)-1]} at epoch {len(avg_accuracy)}")
    #
    print(f'\nfinished fold {fold+1} training')
    print("*" * 30)
#
print(f'\nfinished cross_val model training training')
print("#" * 30)

end = time.time()
print(f'model run time: {end - start}')

