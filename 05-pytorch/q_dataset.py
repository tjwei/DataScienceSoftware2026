from torch.utils.data import Dataset, DataLoader

class MyCIFAR10(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data            # (N, 32, 32, 3) uint8 numpy array
        self.targets = targets      # list of int
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        x = self.data[i]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.targets[i]

ds = MyCIFAR10(train_raw.data, train_raw.targets, train_tf)
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=NUM_WORKERS)
xb, yb = next(iter(loader))
print(xb.shape, xb.dtype, yb.shape, yb[:10])
