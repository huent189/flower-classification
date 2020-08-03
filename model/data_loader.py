from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torch
import os
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

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
    datasets = ImageFolder(root = data_dir, transform=transform)
    dataloaders = {}
    split_len = [int(ratito * len(datasets)) for ratito in split_ratito]
    split_len[-1] = len(datasets) - split_len[0] - split_len[1]
    train, val, test = torch.utils.data.random_split(datasets, split_len)
    dataloaders['train'] = torch.utils.data.DataLoader(train, shuffle=True, batch_size=params.batch_size, num_workers=params.num_workers, pin_memory=params.cuda)
    dataloaders['val'] = torch.utils.data.DataLoader(val, shuffle=True, batch_size=params.batch_size, num_workers=params.num_workers, pin_memory=params.cuda)
    dataloaders['test'] = torch.utils.data.DataLoader(test, shuffle=True, batch_size=params.batch_size, num_workers=params.num_workers, pin_memory=params.cuda)
    return dataloaders