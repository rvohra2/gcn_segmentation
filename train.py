import torch
import torch.nn.functional as F

def train(model, optimizer, loader, adj, Epochs):
    all_logits = []
    for epoch in range(Epochs):
        loss = 0
        for data in loader:
            #print("data = {}".format(data))
            y = data.y
            y = y[0].type(torch.cuda.LongTensor)
            x = data.x
            x = x.cpu()

            optimizer.zero_grad()
            output = model(x, adj).cuda()
            all_logits.append(output.detach())
            #output = output.transpose(0, 1)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()

            # logits = model(x, adj)
            # all_logits.append(logits.detach())
            # logp = F.log_softmax(logits, 1).cuda()
            # loss = F.nll_loss(logp, y)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
    return all_logits