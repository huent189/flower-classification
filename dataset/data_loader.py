import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from dataset.CustomDataset import CustomDataset

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
    transform_train = transforms.Compose([transforms.RandomResizedCrop((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_val = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    datasets = ImageFolder(root=data_dir)
    dataloaders = {}
    split_len = [int(ratito * len(datasets)) for ratito in split_ratito]
    split_len[-1] = len(datasets) - split_len[0] - split_len[1]
    train_ds, val_ds, test_ds = torch.utils.data.random_split(datasets, split_len)
    train_ds = CustomDataset(train_ds, transform_train)
    val_ds = CustomDataset(val_ds, transform_val)
    test_ds = CustomDataset(test_ds, transform_val)
    dataloaders['train'] = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=params.batch_size, num_workers=params.num_workers, pin_memory=params.cuda)
    dataloaders['val'] = torch.utils.data.DataLoader(val_ds, shuffle=True, batch_size=params.batch_size, num_workers=params.num_workers, pin_memory=params.cuda)
    dataloaders['test'] = torch.utils.data.DataLoader(test_ds, shuffle=True, batch_size=params.batch_size, num_workers=params.num_workers, pin_memory=params.cuda)
    return dataloaders
def compute_batch_norm(transform_list, dl):
    nb_sample = len(dl)
    mean = 0
    std = 0
    for X, y in dl:
        mean += X.mean(dim=[0, 2, 3])
        std += X.std(dim=[0, 2, 3])
    return mean / nb_sample, std / nb_sample
