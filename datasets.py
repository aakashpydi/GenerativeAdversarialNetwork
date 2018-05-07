from torch.utils.data.dataset import Dataset
import torchvision.datasets as torch_datasets
import torchvision.transforms as transforms
import torch
import pandas as pd
import numpy as np
from PIL import Image

class RaviVarmaDataset(Dataset):
    def __init__(self, image_size):
        self.data_info = pd.read_csv('ravi_varma_data.csv', header=None)
        self.image_array = np.asarray(self.data_info.iloc[:, 0])
        self.label_array = np.asarray(self.data_info.iloc[:, 1])

        self.transform = transforms.Compose([transforms.Scale(image_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
        #print self.image_array

    def __getitem__(self, index):
        single_image_name = self.image_array[index]
        pil_image = Image.open(single_image_name)
        pil_image = pil_image.resize((32, 32), Image.ANTIALIAS)
        #pil_image.show()
        transformed_image = self.transform(pil_image)
        #print transformed_image.size()
        single_image_label = self.label_array[index]
        return (transformed_image, single_image_label)

    def __len__(self):
        return len(self.data_info.index)

class BapuDataset(Dataset):
    def __init__(self, image_size):
        self.data_info = pd.read_csv('bapu_data.csv', header=None)
        self.image_array = np.asarray(self.data_info.iloc[:, 0])
        self.label_array = np.asarray(self.data_info.iloc[:, 1])

        self.transform = transforms.Compose([transforms.Scale(image_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
        #print self.image_array

    def __getitem__(self, index):
        single_image_name = self.image_array[index]
        pil_image = Image.open(single_image_name)
        pil_image = pil_image.resize((32, 32), Image.ANTIALIAS)
        #print index
        #pil_image.show()
        transformed_image = self.transform(pil_image)
        #print transformed_image.size()
        single_image_label = self.label_array[index]
        return (transformed_image, single_image_label)

    def __len__(self):
        return len(self.data_info.index)

class CombinedDataset:
    def __init__(self, image_size):
        self.data_info = pd.read_csv('combined_data.csv', header=None)
        self.image_array = np.asarray(self.data_info.iloc[:, 0])
        self.label_array = np.asarray(self.data_info.iloc[:, 1])

        self.transform = transforms.Compose([transforms.Scale(image_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
        #print self.image_array

    def __getitem__(self, index):
        single_image_name = self.image_array[index]
        pil_image = Image.open(single_image_name)
        pil_image = pil_image.resize((32, 32), Image.ANTIALIAS)
        #print index
        #pil_image.show()
        transformed_image = self.transform(pil_image)
        #print transformed_image.size()
        single_image_label = self.label_array[index]
        return (transformed_image, single_image_label)

    def __len__(self):
        return len(self.data_info.index)

class DatasetSetup:
    _worker_count = 4

    def __init__(self, image_size, batch_size):
        print "Initializing dataset object."
        self.batch_size = batch_size
        self.image_size = image_size
        self.transform = transforms.Compose([transforms.Scale(image_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
        self.train_dataset = None
        self.train_data_loader = None
        self.total_batch_count = 0

    def initRaviVarma(self):
        self.train_dataset = RaviVarmaDataset(image_size=64)
        self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = False)
        self.total_batch_count = len(self.train_data_loader)

    def initBapu(self):
        self.train_dataset = BapuDataset(image_size=64)
        self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = False)
        self.total_batch_count = len(self.train_data_loader)

    def initCombined(self):
        self.train_dataset = CombinedDataset(image_size=64)
        self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)
        self.total_batch_count = len(self.train_data_loader)

    def initCIFAR10(self):
        self.train_dataset = torch_datasets.CIFAR10(root = './train_data', download = True, transform = self.transform)
        self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True, num_workers = DatasetSetup._worker_count)
        self.total_batch_count = len(self.train_data_loader)
# test = BapuDataset(image_size=64)
# test.__getitem__(0)
# print test.__len__()

#test = RaviVarmaDataset(image_size=64)
#test.__getitem__(0)
#print test.__len__()
