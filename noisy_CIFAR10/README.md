# Noise-padded CIFAR10
## Usage
The dataset is downloaded automatically through torchvision. 

To start the training, simply run:
```
python noisy_cifar10_task.py [args]
```

Options:
- nhid : hidden size of recurrent net
- epochs : number of epochs
- batch : batch size
- lr : learning rate
- nlayers : number of layers
- dt : step size dt of UnICORNN
- alpha : y controle parameter alpha of UnICORNN
