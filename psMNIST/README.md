# Permuted sequential MNIST
## Usage
The dataset is downloaded automatically through torchvision.

To start the training, simply run:
```
python psMNIST_task.py [args]
```

Options:
- nhid : hidden size of recurrent net
- reduce_point : after how many epochs to reduce the lr
- batch : batch size
- lr : learning rate
- nlayers : number of layers
- dt : step size dt of UnICORNN
- alpha : y controle parameter alpha of UnICORNN

The log of the run with a fixed random seed can be found in the results directory.
Note that in order to increase comparability with other RNN methods, we use the same seed for the random permuation as
in other uRNN/oRNN official code repositories (e.g. expRNN, DTRIV and scoRNN).
