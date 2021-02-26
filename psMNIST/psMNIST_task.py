from torch import nn, optim
import torch
import network
import torch.nn.utils
from pathlib import Path
import utils
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--nhid', type=int, default=256,
                    help='hidden size of recurrent net')
parser.add_argument('--reduce_point', type=int, default=650,
                    help='after how many epochs to reduce the lr')
parser.add_argument('--batch', type=int, default=32,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.00251,
                    help='learning rate')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--dt', type=float, default=0.19,
                    help='step size <dt> of UnICORNN')
parser.add_argument('--alpha', type=float, default=30.65,
                    help='y controle parameter <alpha> of UnICORNN')

args = parser.parse_args()
print(args)

ninp = 1
nout = 10

batch_test = 1000

##  same seed as for instance in uRNN/oRNN codes:
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(5544)
np.random.seed(5544)

perm = torch.randperm(784)
model = network.UnICORNN(ninp, args.nhid, nout, args.dt, args.alpha, args.nlayers).cuda()

train_loader, valid_loader, test_loader = utils.get_data(args.batch,batch_test)

## Define the loss
objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def test(data_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            ## Reshape images for sequence learning:
            data = data.reshape(data.size(0), 1, 784)
            data = data.permute(2, 0, 1)
            data = data[perm, :, :]

            output = model(data.cuda())
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred).cuda()).sum()
    accuracy = 100. * correct / len(data_loader.dataset)
    return accuracy.item()

best_eval = 0.
for epoch in range(int(4*args.reduce_point)):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        ## Reshape images for sequence learning:
        images = images.reshape(images.size(0), 1, 784)
        images = images.permute(2, 0, 1)
        images = images[perm, :, :]

        # Training pass
        optimizer.zero_grad()
        output = model(images.cuda())
        loss = objective(output, labels.cuda())
        loss.backward()
        optimizer.step()

    valid_acc = test(valid_loader)
    test_acc = test(test_loader)
    if (valid_acc > best_eval):
        best_eval = valid_acc
        final_test_acc = test_acc

    if(epoch%20==0):
        Path('result').mkdir(parents=True, exist_ok=True)
        f = open('result/psMNIST_log.txt', 'a')
        if (epoch == 0):
            f.write('## learning rate = ' + str(args.lr) + ', dt = ' + str(args.dt) + ', alpha = ' + str(
                args.alpha) + '\n')
        f.write('eval accuracy: ' + str(round(valid_acc, 2)) + '\n')
        f.close()

    if (epoch + 1) == args.reduce_point:
        args.lr /= 10.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

f = open('result/psMNIST_log.txt', 'a')
f.write('final test accuracy: ' + str(round(final_test_acc, 2)) + '\n')
f.close()
