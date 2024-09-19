import torch
# import pandas as pd
from PIL import Image
# import os
import torchvision.transforms as TF 
# from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import numpy as np


def get_compose_list(img_size):
    '''
    get the compose function list based on img size
    '''
    assert len(img_size) == 3

    compose_list=[]

    if img_size[0] == 1:
        compose_list.append(TF.Grayscale(num_output_channels=img_size[0]))

    compose_list.extend(
        # every setting has the followings
        [TF.ToTensor(),
        TF.Resize((img_size[1],img_size[2]), 
                interpolation=TF.InterpolationMode.BICUBIC, 
                antialias=True),
        TF.Lambda(lambda t: (t * 2) - 1), # Scale between [-1, 1] 
        ]
    )
    return compose_list



class ChestXrayDataset(Dataset):
    def __init__(self, img_data_dir, 
                 ds_name,
                 df_data, 
                 img_size=(1,224,224),
                 augmentation=False, 
                 label='Edema',
                 sensitive_label= 'sex',
                 ):
        self.img_data_dir = img_data_dir
        self.ds_name = ds_name  
        self.df_data = df_data
        self.img_size = img_size
        self.do_augment = augmentation
        self.label = label
        self.sensitive_label = sensitive_label

        if self.sensitive_label == 'sex':
            self.col_name_a = 'sex_label'

        else:
            print('{} not implemented'.format(self.sensitive_label))
            raise NotImplementedError



        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(transforms=[T.RandomAffine(degrees=15, scale=(0.9, 1.1))], p=0.5),
        ])
        self.transform = TF.Compose(get_compose_list(self.img_size))

        self.samples = []

        for idx in tqdm((self.df_data.index), desc='Loading Data'):
            if self.ds_name == 'chexpert':
                col_name_pth = 'path_preproc'
            elif self.ds_name == 'NIH':
                col_name_pth = 'Image Index'
                
            path_preproc_idx = self.df_data.columns.get_loc(col_name_pth)
            img_path = self.img_data_dir + self.df_data.iloc[idx, path_preproc_idx]
            img_label = np.array(self.df_data.loc[idx, self.label.strip()] == 1, dtype='float32')
            sensitive_attribute = np.array(self.df_data.loc[idx, self.col_name_a.strip()] == 1, dtype='float32')
            sample = {'image_path': img_path, 'label': img_label, 'sensitive_attribute': sensitive_attribute}
            self.samples.append(sample)

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, item):
        sample = self.get_sample(item)
        # image = T.ToTensor()(sample['image'])
        image = sample['image']
        label = torch.from_numpy(sample['label'])
        sensitive_attribute = torch.from_numpy(sample['sensitive_attribute'])
        


        if self.do_augment:
            image = self.augment(image)

        return {'image': image, 'label': label, 'sensitive_attribute': sensitive_attribute}

    def get_sample(self, item):
        sample = self.samples[item]

        image = Image.open(sample['image_path']).convert('RGB') #PIL image
        image = self.transform(image)
    

        return {'image': image, 'label': sample['label'], 'sensitive_attribute': sample['sensitive_attribute']}

    def exam_augmentation(self,item):
        assert self.do_augment == True, 'No need for non-augmentation experiments'

        sample = self.get_sample(item) #PIL
        image = T.ToTensor()(sample['image'])

        if self.do_augment:
            image_aug = self.augment(image)

        image_all = torch.cat((image,image_aug),axis= 1)
        assert image_all.shape[1]==self.image_size[0]*2, 'image_all.shape[1] = {}'.format(image_all.shape[1])
        return image_all


