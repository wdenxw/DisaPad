"""
Preprocess the target domain data and compute the mmd loss.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
import mmd

NO_CUDA = False
cuda = not NO_CUDA and torch.cuda.is_available()

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

NO_CUDA = False
cuda = not NO_CUDA and torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


def transform_np(x_np_data, y_np_data, batch_size):
    x_tensor = torch.from_numpy(x_np_data)
    y_tensor = torch.from_numpy(y_np_data)
    x_tensor = x_tensor.float()
    y_tensor = y_tensor.float()
    data_train = DealDataset(x_tensor, y_tensor)
    loader = DataLoader(data_train, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
    return loader


def cal_mmd_loss(a_mmd, b_mmd):
    loss = mmd.mmd_rbf_noaccelerate(a_mmd, b_mmd)
    return loss


class DealDataset(Dataset):
    """
    Preprocess the target domain data.
    """

    def __init__(self, f_x, f_y):
        self.X = f_x
        self.Y = f_y
        self.len = self.Y.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len
