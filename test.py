import numpy as np
import colorsys
import torch
import torch.nn.functional as F
import shutil
from matplotlib import pyplot as plt
from convert_svg import render_svg
from utils import select_mask_color_test

def test(model, loader, output_dir, idx, num_instance_label):
    # color
    num_colors = num_instance_label

    correct = 0
    cnt = 0
    iou = []
    with torch.no_grad(): 
        for data in loader:
            y = data.y
            colors = np.array([colorsys.hsv_to_rgb(h, 0.8, 0.8)
                for h in np.linspace(0, 1, 30)]) * 255
            masks = []
            node_num = len(np.unique(data.segmentation))            

            mask = np.zeros((128, 128, 3), np.uint8)
            y_s = y.type(torch.cuda.LongTensor)
            y_s = y_s.unsqueeze(0)  
            print(len(torch.unique(y_s)))
            #print('y_s: ', y_s.min(), y_s.max())

            logits = model(data).cuda()
            logits = logits.unsqueeze(0)
            #print('logits: ', logits.min(), logits.max())
            logp = F.log_softmax(logits, dim=2)

            #print('logp: ', logp.min(), logp.max())
            pred = logp.max(2, keepdim=True)[1]
            
            print(pred.min(), pred.max())
            #print(pred.size())
            for v in range(0, node_num):
                cls = pred[0][v][0].cpu().detach().numpy()
                mask_color = select_mask_color_test(cls, colors)
                
                
                mask[data.segmentation == v] = mask_color
                
            correct += pred.eq(y_s.view_as(pred)).sum().item()
            

            #print(torch.equal(y_s.unsqueeze(1), pred))

            intersect = (y_s.unsqueeze(2) & pred).sum()
            union = (y_s.unsqueeze(2) | pred).sum()
            result = intersect / union
            iou.append(result)
            #print('IOU: ', iou)
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
            cnt += 1
            shutil.move(str(old_svg), str(new_svg))
            #print(new_svg)
        print('Final IOU: ', (sum(iou) / len(iou)))
        #data_num = len(y)  
        #print('\n Accuracy : {}/{} ({:.0f}%)\n'.format(correct,data_num, 100. * correct / data_num))
        return mask