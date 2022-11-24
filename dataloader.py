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
        # lst1 = [8087, 13362, 20582, 32763, 37072, 38056, 42898, 43766, 43926, 44400, 44688, 45073, 48954, 50199, 50637, 55652, 56537, 58794, 63413, 63589, 64030, 65453, 66728, 68055, 68926, 68992, 69521, 69660, 69794, 72424, 72735, 74894, 74948, 75528, 76382, 78257, 78326, 79324, 80217, 80964, 82245, 83732, 84779, 85301, 86866, 87410]
        # lst2 = [1213, 2614, 3035, 3865, 4188, 4243, 4469, 4660, 4727, 5562, 5690, 7406, 9874, 11732, 14322, 15110, 18068, 18816, 19686, 20496, 21635]
        # if self.split == 'train' and idx in lst1:
        #     pass
        # elif self.split == 'val' and idx in lst2:
        #     pass
        
        data = torch.load(osp.join(self.data_folder, f'data_{idx}.pt'))
        return data
            