from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def get_data(batch_size):
    train_data = np.load('data/trainx.npy')
    train_labels = np.load('data/trainy.npy')
    test_data = np.load('data/testx.npy')
    test_labels = np.load('data/testy.npy')
    valid_data = np.load('data/validx.npy')
    valid_labels = np.load('data/validy.npy')

    ## Train data:
    train_dataset = TensorDataset(Tensor(train_data).float(), Tensor(train_labels).long())
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    ## Test data
    test_dataset = TensorDataset(Tensor(test_data).float(), Tensor(test_labels).long())
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=test_labels.size)

    ## Valid data
    valid_dataset = TensorDataset(Tensor(valid_data).float(), Tensor(valid_labels).long())
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=valid_labels.size)

    return train_loader, valid_loader, test_loader