import torch.utils.data as data
import torch
from torchvision import transforms
import pdb 
from scipy.ndimage import imread
import os
from scipy.io import loadmat
import glob
import pickle as pkl
import logging
import argparse


class DCGAN_DataLoader(data.Dataset):
    def __init__(self, root, flag, transform=None, train=True):
        self.train = train
        #data_loader_logger.info('Data_loader is created')
        if self.train:
            self.train_set_path = self.make_dataset_path(root, flag)

    def reader(self, file_path):
       
        with open(file_path, 'rb') as f:
            data = pkl.load(f)
        return torch.from_numpy(data).float()

    def __getitem__(self, index):
        if self.train:
            label_path, condition_path = self.train_set_path[index]
            
            label = self.reader(label_path)
            condition = self.reader(condition_path)
            
            #img = transforms.ToTensor(img)
            #gt = transforms.ToTensor(gt)
            
           
        return label, condition

    def __len__(self):
        return len(self.train_set_path)

    def make_dataset_path(self, root, flag):
        """
        @ reading the own dataset
        """
        dataset_path = []
    
        if flag=='Train': 
            dir_ground_truth = os.path.join(root, 'Train_Data/Label/')
            dir_train_img = os.path.join(root,'Train_Data/Condition/')
            #data_loader_logger.info('Cretating the Training data file path')
        elif flag=='Test':
            dir_ground_truth = os.path.join(root, 'Test_Data/Label/')
            dir_train_img = os.path.join(root,'Test_Data/Condition/')
            #data_loader_logger.info('Cretating the Testing data file path')

        elif flag=='Validation':
            dir_ground_truth = os.path.join(root, r'Validation_Data/Label/')
            dir_train_img = os.path.join(root,r'Validation_Data/Condition/')
            #data_loader_logger.info('Cretating the Validation data file path')
        
        ground_truth_list = os.listdir(dir_ground_truth)
        ground_truth_list.sort()
#        with open('../Data/Dataloader_ground_truth_list.pkl','wb') as f:
#            pkl.dump(ground_truth_list,f)

        for i in ground_truth_list:
            dataset_path.append([dir_ground_truth+i, dir_train_img+i])
          
        #ground_truth_list = [dir_ground_truth+i for i in ground_truth_list]
        #train_img_list = os.listdir(dir_train_img)
        #train_img_list = [dir_train_img +i for i in train_img_list]
        #for gt_path, img_path  in zip(ground_truth_list, train_img_list):
        #    dataset_path.append([img_path, gt_path]
        
        return dataset_path

