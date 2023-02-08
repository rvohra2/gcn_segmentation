import torch
import os.path as osp
import os
from torch_geometric.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.data_file_list = os.listdir(data_folder)

    def __len__(self):
        return len(self.data_file_list)


        

    def __getitem__(self, idx):
        file_name = self.data_file_list[idx]
        data = torch.load(osp.join(self.data_folder, file_name))
        return data
            