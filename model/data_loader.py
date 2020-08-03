from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torch
import os

def fetch_dataloader(data_dir, split_ratito, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.
    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters
    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    transform_list = [transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    datasets = ImageFolder(root = data_dir, transform=transforms.Compose(transform_list))
    dataloaders = {}
    split_len = [int(ratito * len(datasets)) for ratito in split_ratito]
    split_len[-1] = len(datasets) - split_len[0] - split_len[1]
    train, val, test = torch.utils.data.random_split(datasets, split_len)
    dataloaders['train'] = torch.utils.data.DataLoader(train, shuffle=True, batch_size=params.batch_size, num_workers=params.num_workers, pin_memory=params.cuda)
    dataloaders['val'] = torch.utils.data.DataLoader(val, shuffle=True, batch_size=params.batch_size, num_workers=params.num_workers, pin_memory=params.cuda)
    dataloaders['test'] = torch.utils.data.DataLoader(test, shuffle=True, batch_size=params.batch_size, num_workers=params.num_workers, pin_memory=params.cuda)
    return dataloaders
def compute_batch_norm(transform_list, dl):
    nb_sample = len(dl)
    mean = 0
    std = 0
    for X, y in dl:
        mean += X.mean(dim=[0,2,3])
        std += X.std(dim=[0,2,3])
    return mean / nb_sample, std / nb_sample