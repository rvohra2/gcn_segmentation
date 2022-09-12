import numpy as np
import colorsys
import torch

def test(model, adj, num_instance_label, max_dim):
    
    # color
    num_colors = num_instance_label
    colors = np.array([colorsys.hsv_to_rgb(h, 0.8, 0.8)
                   for h in np.linspace(0, 1, num_colors)]) * 255
    mask = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    node_num = len(np.unique(segmentation))
    correct = 0
    with torch.no_grad(): 
        for data in loader:
            y = data.y
            y = y[0].type(torch.cuda.LongTensor)
            x = data.x
            x = x.cpu()

            logits = model(x, adj)
            logp = F.log_softmax(logits, 1)
            pred = logp.max(1, keepdim=True)[1].cuda()
            for v in range(0, node_num):
                cls = pred[v][0].cpu().detach().numpy()
                mask_color = select_mask_color_test(cls, colors)
                
                mask[segmentation == v] = mask_color
            correct += pred.eq(y.view_as(pred)).sum().item()
        data_num = len(y)  
        print('\n Accuracy : {}/{} ({:.0f}%)\n'.format(correct,
                                                   data_num, 100. * correct / data_num))
        return image, mask, target_mask