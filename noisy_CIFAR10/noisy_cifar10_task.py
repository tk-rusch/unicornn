from torch import nn, optim
import torch
import network
import torch.nn.utils
from pathlib import Path
import utils
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size of recurrent net')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs')
parser.add_argument('--batch', type=int, default=30,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.00314,
                    help='learning rate')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--dt', type=float, default=0.126,
                    help='step size <dt> of UnICORNN')
parser.add_argument('--alpha', type=float, default=13.0,
                    help='y controle parameter <alpha> of UnICORNN')

args = parser.parse_args()
print(args)

ninp = 96
nout = 10
bs_test = 1000

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(12345)
np.random.seed(12345)

model = network.UnICORNN(ninp, args.nhid, nout, args.dt, args.alpha, args.nlayers).cuda()
train_loader, valid_loader, test_loader = utils.get_data(args.batch,bs_test)

rands = torch.randn(1, 1000 - 32, 96)
rand_train = rands.repeat(args.batch, 1, 1).cuda()
rand_test = rands.repeat(bs_test, 1, 1).cuda()

## Define the loss
objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def test(data_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = torch.cat((data.cuda().permute(0,2,1,3).reshape(bs_test,32,96),rand_test),dim=1).permute(1,0,2)
            output = model(data.cuda())
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred).cuda()).sum()
    accuracy = 100. * correct / len(data_loader.dataset)
    return accuracy.item()

best_eval = 0.
for epoch in range(args.epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = torch.cat((images.cuda().permute(0,2,1,3).reshape(args.batch,32,96),rand_train),dim=1).permute(1,0,2)
        # Training pass
        optimizer.zero_grad()
        output = model(images.cuda())
        loss = objective(output, labels.cuda())
        loss.backward()
        optimizer.step()
    valid_acc = test(valid_loader)
    test_acc = test(test_loader)
    if(valid_acc > best_eval):
        best_eval = valid_acc
        final_test_acc = test_acc

    Path('result').mkdir(parents=True, exist_ok=True)
    f = open('result/noisy_cifar10_log.txt', 'a')
    if (epoch == 0):
        f.write('## learning rate = ' + str(args.lr) + ', dt = ' + str(args.dt) + ', alpha = ' + str(
            args.alpha) + '\n')
    f.write('eval accuracy: ' + str(round(valid_acc, 2)) + '\n')
    f.close()

    if (epoch + 1) % 250 == 0:
        args.lr /= 10.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

f = open('result/noisy_cifar10_log.txt', 'a')
f.write('final test accuracy: ' + str(round(final_test_acc, 2)) + '\n')
f.close()
