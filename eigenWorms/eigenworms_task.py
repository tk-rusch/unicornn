from torch import nn, optim
import network
import torch
from pathlib import Path
import utils
import argparse

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--nhid', type=int, default=32,
                    help='hidden size of recurrent net')
parser.add_argument('--epochs', type=int, default=250,
                    help='number of epochs')
parser.add_argument('--batch', type=int, default=8,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0076,
                    help='learning rate')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--dt', default=[2.81e-05, 0.0343],
                    help='step size <dt> of UnICORNN')
parser.add_argument('--alpha', type=float, default=0.,
                    help='y controle parameter <alpha> of UnICORNN')

args = parser.parse_args()
print(args)

ninp = 6
nout = 5

trainloader, validloader, testloader = utils.get_data(args.batch)
model = network.UnICORNN(ninp, args.nhid, nout, args.dt, args.alpha, args.nlayers).cuda()
objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def test(dataloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, label in dataloader:
            data = data.permute(1, 0, 2)
            output = model(data.cuda())
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(label.data.view_as(pred).cuda()).sum()
    accuracy = 100. * correct / len(dataloader.dataset)
    return accuracy.item()

best_eval = 0.
for epoch in range(args.epochs):
    model.train()
    for i, (data, label) in enumerate(trainloader):
        optimizer.zero_grad()
        data = data.permute(1, 0, 2)
        output = model(data.cuda())
        loss = objective(output, label.cuda())
        loss.backward()
        optimizer.step()

    valid_acc = test(validloader)
    test_acc = test(testloader)
    if (valid_acc > best_eval):
        best_eval = valid_acc
        final_test_acc = test_acc

    Path('result').mkdir(parents=True, exist_ok=True)
    f = open('result/eigenworms_log.txt', 'a')
    if (epoch == 0):
        f.write('## learning rate = ' + str(args.lr) + ', dt = ' + str(args.dt) + ', alpha = ' + str(
            args.alpha) + '\n')
    f.write('eval accuracy: ' + str(round(valid_acc, 2)) + '\n')
    f.close()

f = open('result/eigenworms_log.txt', 'a')
f.write('final test accuracy: ' + str(round(final_test_acc, 2)) + '\n')
f.close()

