import numpy as np
import colorsys
import torch
import torch.nn.functional as F
import shutil
from matplotlib import pyplot as plt
from convert_svg import render_svg
from utils import select_mask_color
import config
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import utils
import os
import math

def test(model, loader, output_dir, device):
    # color

    cnt = 0
    accuracy_lst = []
    precision_lst = []
    recall_lst = []
    f1_lst = []
    iou = []

    with torch.no_grad():
        print(len(loader))

        for data in loader:
            cnt+=1
            print(cnt)

            # colors = np.array([colorsys.hsv_to_rgb(h, 0.8, 0.8)
            #     for h in np.linspace(0, 1, config.OUTPUT_LAYER)]) * 255
            masks = []
            node_num = len(np.unique(data.segmentation[0]))
            mask = np.zeros((config.INPUT_WDT, config.INPUT_HGT, 3), np.uint8)

            data = data.to(device)
            y = data.y
            
            #y = y.unsqueeze(0)
            
            #y_s = y.type(torch.cuda.LongTensor)
            #print(len(torch.unique(y_s)))
            #print('y_s: ', y_s.min(), y_s.max())

            logits = model(data)
            #logits = logits.unsqueeze(0)
            #print('logits: ', logits.min(), logits.max())
            #logp = F.log_softmax(logits, dim=2)
            #print('logp: ', logp.min(), logp.max())
            #pred = logits.max(2)[1].cuda()
            logp = F.log_softmax(logits, 1)
            pred = logp.max(1)[1]
            #print(pred.min(), pred.max())
            #print(pred.size())
            for v in range(0, node_num):
                cls = pred[v].cpu().detach().numpy()
                mask_color = utils.select_mask_color(cls)
                
                
                mask[data.segmentation[0] == v] = mask_color
            plt.imsave(output_dir / "{:.2f}.png".format(cnt), mask)   
            correct = pred.eq(y.view_as(pred)).sum().item()
            accuracy_lst.append(correct)
            print('Accuracy: ', correct / len(y))
            #print('Report: ', classification_report(y_s, pred))
            
            
            #print(y_s.min(), y_s.max(), pred.min(), pred.max())
            # accuracy: (tp + tn) / (p + n)
            # accuracy = accuracy_score(y.cpu().detach().numpy(), pred.cpu().detach().numpy())
            # accuracy_lst.append(accuracy)
            # # precision tp / (tp + fp)
            # precision = precision_score(y.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='weighted', zero_division=0)
            # precision_lst.append(precision)
            # # recall: tp / (tp + fn)
            # recall = recall_score(y.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='weighted', zero_division=0)
            # recall_lst.append(recall)
            # # f1: 2 tp / (2 tp + fp + fn)
            # f1 = f1_score(y.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='weighted', zero_division=0)
            # f1_lst.append(f1)

            # # confusion matrix
            # matrix = confusion_matrix(y_s, pred)
            # print(matrix)

            #print(torch.equal(y_s.unsqueeze(1), pred))

            intersect = (y & pred).sum()
            union = (y | pred).sum()
            result = intersect / union
            
            if math.isnan(result):
                print('Union: ', union)
                result = intersect / (union+1e-5)

            iou.append(result)
            print('IOU: %f' % result)
            #print('Accuracy: %f' % accuracy, 'Precision: %f' % precision, 'Recall: %f' % recall, 'F1 score: %f' % f1, 'IOU: %f' % result)
            #print(mask.min(), mask.max())

            # plt.imshow(mask)
            # plt.show()

            #print(mask.min(), mask.max())

            colours  = np.unique(mask.reshape(-1,3), axis=0)
            for i,colour in enumerate(colours):
                #print(f'DEBUG: colour {i}: {colour}')
                res = np.where((mask==colour).all(axis=-1),255,0)
                # plt.imshow(res)
                # plt.show()
                res = (res != 0)
                masks.append(res)
                
            #mask = mask.mean(axis=2)
            #print(mask.min(), mask.max())

            


            #mask = (mask != 0)
            #masks.append(mask)
            old_svg = render_svg(masks[1:], node_num)
            new_svg = output_dir / "{:.2f}.svg".format(cnt)
            shutil.move(str(old_svg), str(new_svg))
            #print(new_svg)
        #print('Final Metrics: ', 'Accuracy: ', (sum(accuracy_lst) / len(accuracy_lst)), 'Precision: ', (sum(precision_lst) / len(precision_lst)),
        #'Recall: ', (sum(recall_lst) / len(recall_lst)), 'F1: ', (sum(f1_lst) / len(f1_lst)), 'IOU: ', (sum(iou) / len(iou)))
        print('IOU: ', (sum(iou) / len(iou)))
        print('Accuracy: ', (sum(accuracy_lst) / len(accuracy_lst)))
        #data_num = len(y)  
        #print('\n Accuracy : {}/{} ({:.0f}%)\n'.format(correct,data_num, 100. * correct / data_num))
        return mask


# def test(model, loader, output_dir, num_instance_label):
#     # color
#     num_colors = num_instance_label

#     accuracy = 0
#     cnt = 0
#     iou = []
#     colors = np.array([colorsys.hsv_to_rgb(h, 0.8, 0.8)
#                 for h in np.linspace(0, 1, config.OUTPUT_LAYER)]) * 255
#     colors = torch.from_numpy(colors)
    
#     with torch.no_grad(): 
#         for data in loader:
#             data = data.cuda()
#             y = data.y
            
            
#             masks = []
#             node_num = len(np.unique(data.segmentation))            

#             mask = np.zeros((128, 128, 3), np.uint8)
#             #y_s = y.type(torch.cuda.LongTensor)
#             y_s = y.unsqueeze(0)
#             y = F.one_hot(y_s, config.OUTPUT_LAYER).float()
#             #print(len(torch.unique(y_s)))
#             #print('y_s: ', y_s.min(), y_s.max())

#             logits = model(data)
#             logits = logits.unsqueeze(0)
#             #print('logits: ', logits.min(), logits.max())
#             pred = torch.sigmoid(logits)
#             #print(pred.min(), pred.max())
#             #print(torch.unique(pred))
#             pred[pred >= 0.5] = 1
#             pred[pred < 0.5] = 0

#             #print(pred.size(dim=0),pred.size(dim=1),pred.size(dim=2))
#             #print((pred == y).sum(), (pred == y))
#             accuracy = ((pred == y).sum()/(pred.size(dim=0)*pred.size(dim=1)*pred.size(dim=2))*100)
#             print('Accuracy: ', accuracy)
#             iou.append(accuracy)
#             y = torch.argmax(y, dim=2)
#             print(torch.unique(y))
#             pred = torch.argmax(pred, dim=2)
#             print(torch.unique(pred))

#             #print(pred.size())
#             for v in range(0, node_num):
#                 cls = pred[0][v]
#                 mask_color = select_mask_color_test(cls, colors)
#                 mask[data.segmentation[0] == v] = mask_color
                

#             # plt.imshow(mask)
#             # plt.show()
#             #print(torch.equal(y_s.unsqueeze(1), pred))

#             # intersect = (y_s.unsqueeze(2) & pred).sum()
#             # union = (y_s.unsqueeze(2) | pred).sum()
#             # result = intersect / union
#             # iou.append(result)
#             #print('IOU: ', iou)
#             #print(mask.min(), mask.max())

#             # plt.imshow(mask)
#             # plt.show()

#             #print(mask.min(), mask.max())

#             colours  = np.unique(mask.reshape(-1,3), axis=0)
#             for i,colour in enumerate(colours):
#                 #print(f'DEBUG: colour {i}: {colour}')
#                 res = np.where((mask==colour).all(axis=-1),255,0)
#                 # plt.imshow(res)
#                 # plt.show()
#                 res = (res != 0)
#                 masks.append(res)
                
#             #mask = mask.mean(axis=2)
#             #print(mask.min(), mask.max())

            


#             #mask = (mask != 0)
#             #masks.append(mask)
#             old_svg = render_svg(masks[1:], node_num)
#             new_svg = output_dir / "{:.2f}.svg".format(cnt)
#             cnt += 1
#             shutil.move(str(old_svg), str(new_svg))
#             #print(new_svg)
 
#         print('Final Accuracy: ', (sum(iou) / len(iou)))
#         #data_num = len(y)  
#         #print('\n Accuracy : {}/{} ({:.0f}%)\n'.format(correct,data_num, 100. * correct / data_num))
#         return mask


# def test(model, loader, output_dir, num_instance_label):
#     # color
#     num_colors = num_instance_label

#     correct = 0
#     cnt = 0
#     iou = []
#     with torch.no_grad(): 
#         for data in loader:
#             data = data.cuda()
#             y = data.y
#             colors = np.array([colorsys.hsv_to_rgb(h, 0.8, 0.8)
#                 for h in np.linspace(0, 1, config.OUTPUT_LAYER)]) * 255
#             masks = []
#             node_num = len(np.unique(data.segmentation))            

#             mask = np.zeros((128, 128, 3), np.uint8)
#             #y_s = y.type(torch.cuda.LongTensor)
#             y_s = y.unsqueeze(0)  
#             print(len(torch.unique(y_s)))
#             #print('y_s: ', y_s.min(), y_s.max())

#             logits = model(data)
#             logits = logits.unsqueeze(0)
#             #print('logits: ', logits.min(), logits.max())
#             logp = F.log_softmax(logits, dim=2)

#             #print('logp: ', logp.min(), logp.max())
#             pred = logp.max(2)[1]
            
#             #print(pred.min(), pred.max())
#             #print(pred.size())
#             for v in range(0, node_num):
#                 cls = pred[0][v]
#                 mask_color = select_mask_color_test(cls, colors)
#                 mask[data.segmentation[0] == v] = mask_color
                
                
#             # correct = pred.eq(y_s.view_as(pred))
#             # accuracy = correct.sum().item()
#             # acc = accuracy(pred,y_s)
#             # print('Accuracy: ', acc)
#             # iou.append(acc)
    

#             #print(torch.equal(y_s.unsqueeze(1), pred))

#             # intersect = (y_s & pred).sum()
#             # union = (y_s | pred).sum()
#             # result = intersect / union
#             # result = result[~result.isnan()].mean()
#             # iou.append(result)
#             #print('IOU: ', iou)
#             #print(mask.min(), mask.max())

#             # plt.imshow(mask)
#             # plt.show()

#             #print(mask.min(), mask.max())

#             colours  = np.unique(mask.reshape(-1,3), axis=0)
#             for i,colour in enumerate(colours):
#                 #print(f'DEBUG: colour {i}: {colour}')
#                 res = np.where((mask==colour).all(axis=-1),255,0)
#                 # plt.imshow(res)
#                 # plt.show()
#                 res = (res != 0)
#                 masks.append(res)
                
#             #mask = mask.mean(axis=2)
#             #print(mask.min(), mask.max())
#             print(len(torch.unique(pred)))
#             correct = pred.eq(y_s.view_as(pred)).sum().item()
#             data_num = y_s.size(0)*y_s.size(1)
#             print('\n Accuracy : {}/{} ({:.0f}%)\n'.format(correct,
#                                                    data_num, 100. * correct / data_num))
#             iou.append(correct / data_num)


#             #mask = (mask != 0)
#             #masks.append(mask)
#             old_svg = render_svg(masks[1:], node_num)
#             new_svg = output_dir / "{:.2f}.svg".format(cnt)
#             cnt += 1
#             shutil.move(str(old_svg), str(new_svg))
#             #print(new_svg)
        
#         print('Final IOU: ', (sum(iou) / len(iou)))
#         #data_num = len(y)  
#         #print('\n Accuracy : {}/{} ({:.0f}%)\n'.format(correct,data_num, 100. * correct / data_num))
#         return mask