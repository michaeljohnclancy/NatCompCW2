from torch.utils.data import SubsetRandomSampler
import numpy as np
import torch

def load_tensors(dataloc, shuffle=True, validation_split=0.5, random_seed=12345, phi=lambda x: x, cuda_device=None):
    data = np.loadtxt(dataloc)

    dataset_size = len(data)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    train_data = torch.from_numpy(data[train_indices]).float()
    val_data = torch.from_numpy(data[val_indices]).float()

    if cuda_device is not None:
        x_train = phi(train_data[:, :2]).to(cuda_device)
        y_train = train_data[:, 2:].to(cuda_device).type(torch.FloatTensor).cuda()
        x_val = phi(val_data[:, :2]).to(cuda_device)
        y_val = val_data[:, 2:].to(cuda_device).type(torch.FloatTensor).cuda()
    else:
        x_train = phi(train_data[:, :2])
        y_train = train_data[:, 2:].type(torch.FloatTensor)
        x_val = phi(val_data[:, :2])
        y_val = val_data[:, 2:].type(torch.FloatTensor)

    return x_train, y_train, x_val, y_val


def phi(X):
    sinX = torch.sin(X)
    squaredX = torch.pow(X, 2)
    return torch.hstack([X, sinX, squaredX])

