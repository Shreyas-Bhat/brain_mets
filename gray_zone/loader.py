import os
import pandas as pd
import json
import math
import numpy as np
import torch
import matplotlib.image as mpimg
from monai.transforms import Compose
from sklearn.model_selection import train_test_split

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, image_list, label_list, transform=None):
        self.df = df
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


def loader(data_path: str,
           output_path: str,
           train_transforms: Compose,
           val_transforms: Compose,
           metadata_path: str = None,
           train_frac: float = 0.65,
           test_frac: float = 0.25,
           seed: int = 0,
           batch_size: int = 32,
           balanced: bool = False,
           weights: torch.Tensor = None,
           label_colname: str = 'label',
           image_colname: str = 'image',
           split_colname: str = 'dataset',
           patient_colname: str = 'patient'):
    """
    Inspired by https://github.com/Project-MONAI/tutorials/blob/master/2d_classification/mednist_tutorial.ipynb

    Returns:
        DataLoader, DataLoader, DataLoader, pd.Dataframe: train dataset, validation dataset, val dataset, test dataset,
         test df
    """
    # Load metadata and create val/train/test split if not already done
    split_df = split_dataset(output_path, train_frac=train_frac, test_frac=test_frac,
                             seed=seed, metadata_path=metadata_path, split_colname=split_colname, image_colname=image_colname,
                             patient_colname=patient_colname) # added image_colname on 08/21/2022, remove for DC
    train_loader, val_loader, test_loader = None, None, None
    df_train = split_df[split_df[split_colname] == "train"]
    if len(df_train):
        if balanced:
            sampler, weights = get_balanced_sampler(df_train, label_colname, weights)
        shuffle = not balanced
        train_ds = Dataset(df_train, data_path, train_transforms, label_colname, image_colname)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=10, sampler=sampler if balanced else None)

    df_val = split_df[split_df[split_colname] == "val"]
    if len(df_val):
        if balanced:
            sampler, _ = get_balanced_sampler(df_val, label_colname, weights)
        val_ds = Dataset(df_val, data_path, val_transforms, label_colname, image_colname)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, num_workers=10, sampler=sampler if balanced else None)

    df_test = split_df[split_df[split_colname] == "test"]
    if len(df_test):
        test_ds = Dataset(df_test, data_path, val_transforms, label_colname, image_colname)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=10)
    return train_loader, val_loader, test_loader, df_val, df_test, weights


def get_unbalanced_loader(df, data_path, batch_size, transforms, label_colname, image_colname):
    ds = Dataset(df, data_path, transforms, label_colname, image_colname)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=10, sampler=None)


def get_balanced_sampler(split_df: pd.DataFrame,
                         label_name: str,
                         weights: None):
    """ Balances the sampling of classes to have equal representation. """
    labels, count = np.unique(split_df[label_name], return_counts=True)
    weight_count = (1 / torch.Tensor(count)).float()
    if weights is None:
        weights = weight_count
    else:
        weights *= weight_count
    sample_weights = torch.tensor([weights[int(l)] for l in split_df[label_name]]).float()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights))
    return sampler, weights


def split_dataset(output_path: str,
                  metadata_path: str,
                  train_frac: float,
                  test_frac: float,
                  seed: int,
                  split_colname: str,
                  image_colname: str,
                  patient_colname: str):
    """Load csv file containing metadata (image filenames, labels, patient ids, and val/train/test split)"""
    split_df_path = os.path.join(output_path, "split_df.csv")

    # If output_path / "split_df.csv" exists use the already split csv
    if os.path.isfile(split_df_path):
        df = pd.read_csv(split_df_path)
    # If output_path / "split_df.csv" doesn't exist: split images by patient using the train and test fractions
    else:
        df = pd.read_csv(metadata_path)
        # If images are not already split into val/train/test, split by patient
        if split_colname not in df:
            print('generating splits based on metrics provided')
            patient_lst = list(set(df[patient_colname].tolist()))
            train_patients, remain_patients = train_test_split(patient_lst, train_size=train_frac, random_state=seed)
            test_patients, val_patients = train_test_split(remain_patients, train_size=test_frac / (1 - train_frac),
                                                           random_state=seed)

            df[split_colname] = None
            df.loc[df[patient_colname].isin(train_patients), split_colname] = 'train'
            df.loc[df[patient_colname].isin(val_patients), split_colname] = 'val'
            df.loc[df[patient_colname].isin(test_patients), split_colname] = 'test'

        df.to_csv(split_df_path)

    return df
