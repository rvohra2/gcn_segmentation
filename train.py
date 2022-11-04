import torch
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import gc
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.autograd import Variable

from utils import segmentation_adjacency, create_mask, save_ckp, load_ckp, focal_loss

writer = SummaryWriter()
log_name = '{}_{}'.format('gcn', datetime.now().strftime('%Y%m%d_%H%M%S'))
writer = SummaryWriter('logs/{}'.format(log_name))
def val(model, loader):
    all_logits = []
    model.eval()  
    loss = 0
    with torch.no_grad():
        for data in loader:
            gc.collect()
            torch.cuda.empty_cache()
            #print("data = {}".format(data))
            y = data.y
            y_s = y.type(torch.cuda.LongTensor) 
            y = y_s.unsqueeze(0)        
            logits = model(data).cuda()
            logits = logits.unsqueeze(0)
            #all_logits.append(logits.detach())
            loss1 = F.cross_entropy(logits.permute(0,2,1), y)
            loss2 = focal_loss(logits, y)
            loss = loss1 + loss2


    return loss.item()

def train(model, optimizer, loader):
    all_logits = []

    model.train()
    loss = 0
    train_epoch_loss = 0
    for data in loader:
        gc.collect()
        torch.cuda.empty_cache()

        y = data.y
        y_s = y.type(torch.cuda.LongTensor)
        y = y_s.unsqueeze(0)
        #y_s = y.type(torch.cuda.LongTensor)
        
        logits = model(data).cuda()
        logits = logits.unsqueeze(0)
        #all_logits.append(logits.detach())
        #logp = F.log_softmax(logits, 1)
        # adj = np.array(data.adj)
        # print(adj[0].shape[0])
        #segmentation = data.segmentation
        #img, mask = create_mask(data.image, segmentation[0], adj[0].shape[0], epoch, data.node, all_logits)
        loss1 = F.cross_entropy(logits.permute(0,2,1), y)
        loss2 = focal_loss(logits, y)

        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #return ((train_epoch_loss)/len(loader))
    return loss.item()

def main(model, optimizer, loader, val_loader, start_epoch, Epochs, valid_loss_min):

    for epoch in range(start_epoch, Epochs):
        train_loss = train(model, optimizer, loader)
        val_loss = val(model, val_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', val_loss, epoch)
        print('Epoch %d | Training Loss: %.4f| Validation Loss: %.4f' % (epoch, train_loss, val_loss))

        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': val_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        if epoch%10 == 0:
            # save checkpoint
            save_ckp(checkpoint, False, "/home/rhythm/notebook/vectorData_test/temp/chk.pt", "/home/rhythm/notebook/vectorData_test/temp/model.pt")

        if val_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,val_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, "/home/rhythm/notebook/vectorData_test/temp/chk.pt", "/home/rhythm/notebook/vectorData_test/temp/model.pt")
            valid_loss_min = val_loss
                
    return model