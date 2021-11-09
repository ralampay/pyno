from torch.utils.data import Dataset, DataLoader

class PairedAbstractDataset(Dataset):
    def __init__(self, X, Y):
        self.x = X
        self.y = Y
        self.n_samples = len(X)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
