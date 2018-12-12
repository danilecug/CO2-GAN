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

parser = argparse.ArgumentParser()
#parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs of training')
#parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
#parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
#parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of graident')
#parser.add_argument('--b2', type=float, default=0.9899, help='adam:decay of first order momentum of gradient')
#parser.add_argument('--n_cpu', type=int, default=8, help='number of cup threads to use during batch generation')
#parser.add_argument('--latent_dim', type=int, default=8, help='dimensionality of the latent space')
#parser.add_argument('--image_size', type=int, default=128, help='size of the each image dimension')
#parser.add_argument('--channels', type=int, default=1, help='number of the image channels')
#parser.add_argument('--root', type=str, default='../Data/', help='root of the data')
#parser.add_argument('--d_step', type=int, default= 5, help='discrimator step')
#parser.add_argument('--g_step', type=int, default= 5, help='generate step')
parser.add_argument('--chpt', type=str, default='../Model/Model_Oct_07/', help='training data saving path')

opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.chpt):
    os.mkdir(opt.chpt)
    os.mkdir(opt.chpt+'Model/')

data_loader_logger = logging.getLogger('myloger')
data_loader_logger.setLevel(logging.DEBUG)
# create one handler, writing the log
fh = logging.FileHandler(opt.chpt+ 'DataLoader.log')

# create one handler, print the result in the terminal
ch = logging.StreamHandler()

# define the formate of handlers
formaters = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
fh.setFormatter(formaters)
ch.setFormatter(formaters )

data_loader_logger.addHandler(fh)
data_loader_logger.addHandler(ch)


class DCGAN_DataLoader(data.Dataset):
    def __init__(self, root, flag, transform=None, train=True):
        self.train = train
        data_loader_logger.info('Data_loader is created')
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
            data_loader_logger.info('Cretating the Training data file path')
        elif flag=='Test':
            dir_ground_truth = os.path.join(root, 'Test_Data/Label/')
            dir_train_img = os.path.join(root,'Test_Data/Condition/')
            data_loader_logger.info('Cretating the Testing data file path')

        elif flag=='Validation':
            dir_ground_truth = os.path.join(root, r'Validation_Data/Label/')
            dir_train_img = os.path.join(root,r'Validation_Data/Condition/')
            data_loader_logger.info('Cretating the Validation data file path')

        ground_truth_list = os.listdir(dir_ground_truth)
 
        for i in ground_truth_list:
            dataset_path.append([dir_ground_truth+i, dir_train_img+i])
          
        #ground_truth_list = [dir_ground_truth+i for i in ground_truth_list]
        #train_img_list = os.listdir(dir_train_img)
        #train_img_list = [dir_train_img +i for i in train_img_list]
        #for gt_path, img_path  in zip(ground_truth_list, train_img_list):
        #    dataset_path.append([img_path, gt_path]
        
        return dataset_path

