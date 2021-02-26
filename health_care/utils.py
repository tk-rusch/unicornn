from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def get_data(name, batch_train, batch_test):
    train_data = np.load('data/' + name + '/trainx.npy')
    train_labels = np.load('data/' + name + '/trainy.npy')
    valid_data = np.load('data/' + name + '/validx.npy')
    valid_labels = np.load('data/' + name + '/validy.npy')
    test_data = np.load('data/' + name + '/testx.npy')
    test_labels = np.load('data/' + name + '/testy.npy')

    ## Train data:
    train_dataset = TensorDataset(Tensor(train_data).float(), Tensor(train_labels).float())
    trainloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_train)

    ## Test data
    test_dataset = TensorDataset(Tensor(test_data).float(), Tensor(test_labels).float())
    testloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_test)

    ## Valid data
    valid_dataset = TensorDataset(Tensor(valid_data).float(), Tensor(valid_labels).float())
    validloader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_test)

    return trainloader, train_dataset, validloader, valid_dataset, testloader, test_dataset