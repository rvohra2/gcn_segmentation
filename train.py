import torch
import gc
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from utils import save_ckp, focal_loss, sigmoid_focal_loss
import config
import torch.nn as nn

writer = SummaryWriter()
log_name = '{}_{}'.format('gcn', datetime.now().strftime('%Y%m%d_%H%M%S'))
writer = SummaryWriter('logs/{}'.format(log_name))
def val(model, loader, device):
    model.eval()  
    val_loss = 0
    cnt = 0
    with tqdm(loader, unit="batch") as tepoch:
        with torch.no_grad():
            for data in tepoch:
                #mask = torch.zeros((1,128, 128)).cuda()
                cnt +=1
                data = data.to(device)
                y = data.y
                y = y.long()
                #y = y.unsqueeze(0)      
                logits = model(data)
                ##logits = logits.unsqueeze(0)
                #loss = F.nll_loss(logits, y)
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
                #loss = F.cross_entropy(logits, y)
                t = F.one_hot(y, config.OUTPUT_LAYER).float()
                loss = sigmoid_focal_loss(logits, t)

                # if(config.is_L1==True):
            
                #     l1_crit = nn.L1Loss(size_average=False)
                #     reg_loss = 0
                #     for param in model.parameters():
                #         reg_loss += l1_crit(param,target=torch.zeros_like(param))

                #     factor = 0.00005
                #     loss += factor * reg_loss

                val_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
                writer.add_scalar('Loss/data/val', loss, cnt)


    return ((val_loss)/(len(loader)))

def train(model, optimizer, schedueler, loader, device):
    model.train()
    train_loss = 0
    train_losses = []
    cnt = 0
    with tqdm(loader, unit="batch") as tepoch:
        for data in tepoch:
            optimizer.zero_grad()
            cnt +=1
            #mask = torch.zeros((1,128, 128)).cuda()
            data = data.to(device)
            y = data.y
            y = y.long()
            #y = y.unsqueeze(0)
            logits = model(data)

            ##logits = logits.unsqueeze(0)

            #loss = F.nll_loss(logits, y)
            # logp = F.log_softmax(logits, dim=2)
            # pred = logp.max(2)[1]
            # #print(logits.min(), logits.max(), logp.min(), logp.max())
            # node_num = len(torch.unique(data.segmentation))

            # for v in range(0, node_num):
            #     cls = pred[0][v].float()
            #     #print(cls)
            #     mask[0][data.segmentation == v] = cls
            #loss = F.cross_entropy(logits, y)
            
            # mask = mask/config.OUTPUT_LAYER
            # y = y/config.OUTPUT_LAYER
            #print(mask.min(), mask.max(), torch.unique(mask))
            #print(y.min(), y.max(), torch.unique(y))
            #loss = F.mse_loss(mask.requires_grad_(True), y)
            t = F.one_hot(y, config.OUTPUT_LAYER).float()
            loss = sigmoid_focal_loss(logits, t)

            #Adding code for L1 Regularisation
            # if(config.is_L1==True):
            
            #     l1_crit = nn.L1Loss(size_average=False)
            #     reg_loss = 0
            #     for param in model.parameters():
            #         reg_loss += l1_crit(param,target=torch.zeros_like(param))

            #     factor = 0.00005
            #     loss += factor * reg_loss
                #train_losses.append(loss)

            # else:
            #     train_losses.append(loss)

            # loss = focal_loss(logits, y)
            
            loss.backward()
            optimizer.step()
            #schedueler.step()
            train_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())
            writer.add_scalar('Loss/data/train', loss, cnt)

    return ((train_loss)/len(loader))
    #return loss.item()

def main(model, optimizer, schedueler,loader, val_loader, start_epoch, Epochs, device, valid_loss_min):

    for epoch in range(start_epoch, Epochs):
        gc.collect()
        torch.cuda.empty_cache()
        train_loss = train(model, optimizer, schedueler,loader, device)
        val_loss = val(model, val_loader, device)
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
            save_ckp(checkpoint, False, config.CHK_PATH, config.MODEL_PATH)

        if val_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,val_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, config.CHK_PATH, config.MODEL_PATH)
            valid_loss_min = val_loss
                
    return model