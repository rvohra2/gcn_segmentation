import torch
import gc
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from utils import save_ckp, sigmoid_focal_loss
import config

writer = SummaryWriter()
log_name = '{}_{}'.format('gcn', datetime.now().strftime('%Y%m%d_%H%M%S'))
writer = SummaryWriter('logs/{}'.format(log_name))
def val(model, loader):
    model.eval()  
    val_loss = 0
    cnt = 0
    with tqdm(loader, unit="batch") as tepoch:
        with torch.no_grad():
            for data in tepoch:
                mask = torch.zeros((1,128, 128)).cuda()
                data = data.cuda()
                cnt +=1
                y = data.y
                y = y.unsqueeze(0).float()        
                logits = model(data)
                logits = logits.unsqueeze(0)
                # logp = F.log_softmax(logits, dim=2)
                # pred = logp.max(2)[1]
                # node_num = len(torch.unique(data.segmentation))
                # for v in range(0, node_num):
                #     cls = pred[0][v].float()
                #     mask[0][data.segmentation == v] = cls
                # mask = mask/config.OUTPUT_LAYER
                # y = y/config.OUTPUT_LAYER
                # loss = F.mse_loss(mask, y)
                #val_loss += loss
                loss = F.cross_entropy(logits.permute(0,2,1), y)
                #t = F.one_hot(y, config.OUTPUT_LAYER).float()
                #loss = sigmoid_focal_loss(logits, t)
                tepoch.set_postfix(loss=loss.item())
                writer.add_scalar('Loss/data/val', loss, cnt)


    return loss.item()

def train(model, optimizer, loader):
    model.train()
    train_loss = 0
    cnt = 0
    with tqdm(loader, unit="batch") as tepoch:
        for data in tepoch:
            mask = torch.zeros((1,128, 128)).cuda()
            data = data.cuda()
            cnt +=1
            y = data.y
            y = y.unsqueeze(0).float()
            logits = model(data)
            logits = logits.unsqueeze(0)
            # logp = F.log_softmax(logits, dim=2)
            # pred = logp.max(2)[1]
            # #print(logits.min(), logits.max(), logp.min(), logp.max())
            # node_num = len(torch.unique(data.segmentation))

            # for v in range(0, node_num):
            #     cls = pred[0][v].float()
            #     #print(cls)
            #     mask[0][data.segmentation == v] = cls
            loss = F.cross_entropy(logits.permute(0,2,1), y)
            
            # mask = mask/config.OUTPUT_LAYER
            # y = y/config.OUTPUT_LAYER
            #print(mask.min(), mask.max(), torch.unique(mask))
            #print(y.min(), y.max(), torch.unique(y))
            #loss = F.mse_loss(mask.requires_grad_(True), y)
            # t = F.one_hot(y, config.OUTPUT_LAYER).float()
            # loss = sigmoid_focal_loss(logits, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #train_loss += loss
            tepoch.set_postfix(loss=loss.item())
            writer.add_scalar('Loss/data/train', loss, cnt)

    #return ((train_epoch_loss)/len(loader))
    return loss.item()

def main(model, optimizer, loader, val_loader, start_epoch, Epochs, valid_loss_min):

    for epoch in range(start_epoch, Epochs):
        gc.collect()
        torch.cuda.empty_cache()
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