from torch import nn, optim
import utils
import network
import torch
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--name', default='RR',
                    help='which dataset: HR or RR')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size of recurrent net')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs')
parser.add_argument('--batch', type=int, default=32,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.00398,
                    help='learning rate')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--dt', default=0.011,
                    help='step size <dt> of UnICORNN')
parser.add_argument('--alpha', type=float, default=9.0,
                    help='y controle parameter <alpha> of UnICORNN')

args = parser.parse_args()
print(args)

ninp = 2
nout = 1
batch_test = 100

trainloader, train_dataset, validloader, valid_dataset, testloader, test_dataset = utils.get_data(args.name,args.batch,batch_test)
model = network.UnICORNN(ninp, args.nhid, nout, args.dt, args.alpha, args.nlayers).cuda()

objective = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
objective_test = nn.MSELoss(reduction='sum')

def test(dataloader,dataset):
    model.eval()
    loss = 0.
    with torch.no_grad():
        for i, (data, label) in enumerate(dataloader):
            data = data.permute(1, 0, 2)
            output = model(data.cuda()).squeeze(-1)
            loss += objective_test(output, label.cuda())
        loss /= len(dataset)
        loss = torch.sqrt(loss)
    return loss.item()

best_val_loss = 100000.
for epoch in range(args.epochs):
    model.train()
    for i, (data, label) in enumerate(trainloader):
        optimizer.zero_grad()
        data = data.permute(1, 0, 2)
        output = model(data.cuda()).squeeze(-1)
        loss = objective(output, label.cuda())
        loss.backward()
        optimizer.step()

    valid_loss = test(validloader,valid_dataset)
    test_loss = test(testloader,test_dataset)
    if(valid_loss<best_val_loss):
        best_val_loss = valid_loss
        final_test_loss = test_loss

    Path('results').mkdir(parents=True, exist_ok=True)
    f = open('results/'+args.name+'_log.txt', 'a')
    if (epoch == 0):
        f.write('## learning rate = ' + str(args.lr) + ', dt = ' + str(args.dt) + ', alpha = ' + str(
            args.alpha) + '\n')
    f.write('eval loss: ' + str(round(valid_loss, 2)) + '\n')
    f.close()

    if (epoch + 1) == 250:
        args.lr /= 10.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

f = open('results/'+args.name+'_log.txt', 'a')
f.write('final test loss: ' + str(round(final_test_loss, 2)) + '\n')
f.close()
