from torch import nn, optim
import torch
import network
import torch.nn.utils
from pathlib import Path
import utils
import argparse

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size of recurrent net')
parser.add_argument('--emb_dim', type=int, default=100,
                    help='embedding size')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs')
parser.add_argument('--batch', type=int, default=32,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.000164,
                    help='learning rate')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--drop', type=float, default=0.61,
                    help='variational dropout')
parser.add_argument('--drop_emb', type=float, default=0.65,
                    help='embedding dropout')
parser.add_argument('--dt', default=[0.0066, 0.205],
                    help='step size <dt> of UnICORNN')
parser.add_argument('--alpha', type=float, default=0.,
                    help='y controle parameter <alpha> of UnICORNN')

args = parser.parse_args()
print(args)

## set up data iterators and dictionary:
train_iterator, valid_iterator, test_iterator, text_field = utils.get_data(args.batch,args.emb_dim)

ninp = len(text_field.vocab)
nout = 1
pad_idx = text_field.vocab.stoi[text_field.pad_token]

model = network.UnICORNN(ninp, args.emb_dim, args.nhid, nout, pad_idx,
                         args.dt, args.alpha, args.nlayers, args.drop,
                         args.drop_emb).cuda()

## zero embedding for <unk_token> and <padding_token>:
utils.zero_words_in_embedding(model,args.emb_dim,text_field,pad_idx)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.BCEWithLogitsLoss()
print('done building')

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        text = batch.text
        predictions = model(text.cuda()).squeeze(1)
        loss = criterion(predictions, batch.label.cuda())
        acc = binary_accuracy(predictions, batch.label.cuda())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text = batch.text
            predictions = model(text.cuda()).squeeze(1)
            loss = criterion(predictions, batch.label.cuda())
            acc = binary_accuracy(predictions, batch.label.cuda())
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator) * 100.

best_eval = 0.
for epoch in range(args.epochs):
    train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    if (valid_acc > best_eval):
        best_eval = valid_acc
        final_test_acc = test_acc

    Path('result').mkdir(parents=True, exist_ok=True)
    f = open('result/IMDB_log.txt', 'a')
    if (epoch == 0):
        f.write('## learning rate = ' + str(args.lr) + ', dt = ' + str(args.dt) + ', alpha = ' + str(
            args.alpha) + '\n')
    f.write('eval accuracy: ' + str(round(valid_acc, 2)) + '\n')
    f.close()

f = open('result/IMDB_log.txt', 'a')
f.write('final test accuracy: ' + str(round(final_test_acc, 2)) + '\n')
f.close()
