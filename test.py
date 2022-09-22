import numpy as np
import colorsys
import torch
import torch.nn.functional as F
import shutil
from matplotlib import pyplot as plt
from convert_svg import render_svg
from utils import select_mask_color_test

def test(model, loader, output_dir, ids, idx, num_instance_label):

    # color
    num_colors = num_instance_label
    colors = np.array([colorsys.hsv_to_rgb(h, 0.8, 0.8)
                for h in np.linspace(0, 1, num_colors)]) * 255
    
    
    correct = 0
    cnt = 0
    iou = []
    with torch.no_grad(): 
        for data in loader:
            masks = []
            node_num = len(np.unique(data.segmentation))
            y = data.y

            mask = np.zeros((64, 64, 3), np.uint8)
            y_s = y.type(torch.cuda.LongTensor)

            logits = model(data)
            logp = F.log_softmax(logits, 1)
            pred = logp.max(1, keepdim=True)[1].cuda()
            for v in range(0, node_num):
                cls = pred[v][0].cpu().detach().numpy()
                mask_color = select_mask_color_test(cls, colors)
                
                mask[data.segmentation[0] == v] = mask_color
                
            correct += pred.eq(y_s.view_as(pred)).sum().item()
            

            #print(torch.equal(y_s.unsqueeze(1), pred))

            intersect = (y_s.unsqueeze(1) & pred).sum()
            union = (y_s.unsqueeze(1) | pred).sum()
            result = intersect / union
            iou.append(result)
            #print('IOU: ', iou)

            mask = mask.mean(axis=2)
            mask = (mask != 0)
            masks.append(mask)
            old_svg = render_svg(masks, node_num)
            new_svg = output_dir / "{:s}.svg".format(ids[cnt])
            cnt += 1
            shutil.move(str(old_svg), str(new_svg))
        print('Final IOU: ', (sum(iou) / len(iou)))
        #data_num = len(y)  
        #print('\n Accuracy : {}/{} ({:.0f}%)\n'.format(correct,data_num, 100. * correct / data_num))
        return mask