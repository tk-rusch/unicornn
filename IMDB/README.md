# IMDB 

## Overview
The IMDB data set is a collection of written movie reviews. The aim of this binary sentiment classification task is to decide whether a movie review is positive or negative.
## Data
The data set consists of 50k movie reviews. 25k reviews are used for training 
(with 7.5k of them are used for evaluating) and 25k reviews are used for testing.
## Usage
The dataset is downloaded automatically through torchtext.

To start the training, simply run:
```
python IMDB_task.py [args]
```

Options:
- nhid : hidden size of recurrent net
- emb_dim : embedding size
- epochs : number of epochs
- batch : batch size
- lr : learning rate
- nlayers : number of layers
- drop : variational dropout
- drop_emb : embedding dropout
- dt : step size dt of UnICORNN
- alpha : y controle parameter alpha of UnICORNN
