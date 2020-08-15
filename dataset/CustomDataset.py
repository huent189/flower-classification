from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.dataset[index][0])
        else:
            x = self.dataset[index][0]
        return x, self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)
