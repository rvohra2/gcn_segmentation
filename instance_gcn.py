import argparse
import torch.optim as optim
from torch_geometric.loader import DataLoader
import os
import numpy as np
from pathlib import Path
from dataloader import MyDataset
from dataset import get_dataset
from train import main
from test import test
from gcn_model import GCNs
from utils import load_ckp
import config
import shutil
 

start_epoch = 0
valid_loss_min = np.inf


if config.SPLIT == 'train':
    # with (Path(config.DATASET_PATH) / config.DATASET/ 'lst.txt').open("r") as f:
    #             ids = [_.strip() for _ in f.readlines()]

    # np.random.shuffle(ids)
    # train_data, val_data = np.split(np.array(ids), [int(len(ids)*0.9)])

    # with open(Path(config.DATASET_PATH) / config.DATASET/ "train.txt", 'w') as f:
    #     for line in train_data:
    #         f.write(f"{line}\n")

    # with open(Path(config.DATASET_PATH) / config.DATASET/ "val.txt", 'w') as f:
    #     for line in val_data:
    #         f.write(f"{line}\n")

    dataset_train = get_dataset(config.DATASET, config.SPLIT, config.TRANSFORM)
    lst_train = os.listdir(Path(config.ROOT_PATH) / 'train' )
    if len(lst_train) != len(dataset_train):
        for idx in range(len(dataset_train)): 
            data_train = dataset_train[idx]

    train_ds = MyDataset(Path(config.ROOT_PATH) / 'train')
    loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, num_workers=0, drop_last=True)

    dataset_val = get_dataset(config.DATASET, 'val', False)
    lst_val = os.listdir(Path(config.ROOT_PATH) / 'val')
    if len(lst_val) != len(dataset_val):
        for idx in range(len(dataset_val)): 
            data_val = dataset_val[idx]

    val_ds = MyDataset(Path(config.ROOT_PATH) / 'val')
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, num_workers=0, drop_last=True)

else:
    dataset_test = get_dataset(config.DATASET, config.SPLIT, False)
    lst_test = os.listdir(Path(config.ROOT_PATH) / 'test')
    if len(lst_test) != len(dataset_test):
        for idx in range(len(dataset_test)): 
            data_test = dataset_test[idx]

    test_ds = MyDataset(Path(config.ROOT_PATH) / 'test')
    loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE)


print('Dataset loaded successfully')


# model = GCNs(loader.dataset[0].x.shape[1], 1024, 30)
# optimizer = optim.Adam(model.parameters(), lr)
# all_logits = main(model, optimizer, loader, val_loader, start_epoch, Epochs, valid_loss_min)


model = GCNs(loader.dataset[0].x.shape[1], config.HIDDEN_LAYER, config.OUTPUT_LAYER).cuda()
optimizer = optim.Adam(model.parameters(), config.LR)
if config.SPLIT == 'train':
    # if os.path.exists(config.CHK_PATH):
    # model, optimizer, start_epoch, valid_loss_min = load_ckp(config.CHK_PATH, model, optimizer)
    # print("model = ", model)
    # print("optimizer = ", optimizer)
    # print("start_epoch = ", start_epoch)
    # print("valid_loss_min = ", valid_loss_min)
    # print("valid_loss_min = {:.6f}".format(valid_loss_min))

    all_logits = main(model, optimizer, loader, val_loader, start_epoch, config.EPOCHS, valid_loss_min)
else:
    model, optimizer, start_epoch, valid_loss_min = load_ckp(config.CHK_PATH, model, optimizer)
    model.eval()
    output_dir = Path(config.OUTPUT_PATH)
    output_dir.mkdir(exist_ok=True, parents=True)
    mask = test(model, loader, output_dir, loader.dataset[0].y)




    # model, optimizer, start_epoch, valid_loss_min = load_ckp(ckp_path, model, optimizer)
    # print("model = ", model)
    # print("optimizer = ", optimizer)
    # print("start_epoch = ", start_epoch)
    # print("valid_loss_min = ", valid_loss_min)
    # print("valid_loss_min = {:.6f}".format(valid_loss_min))
    


    

