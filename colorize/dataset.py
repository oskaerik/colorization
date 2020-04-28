import torch
import numpy as np
from skimage import transform
from torch.utils import data
from . import util

class Dataset(data.Dataset):
    """Dataset for training the PyTorch model."""

    def __init__(self, paths, out_size=(56, 56)):
        """Constructor setting the paths."""
        self.paths = paths
        self.out_size = out_size

    def __len__(self):
        """Returns the number of samples."""
        return len(self.paths)

    def __getitem__(self, index):
        """Returns a sample."""
        try:
            # Load image
            X, Y = util.imread(self.paths[index])

            # Normalize X between -1 and 1
            X = (X - 50) / 50

            # Resize and encode
            Y = transform.resize(Y, self.out_size)
            Z = util.soft_encode(Y)

            # Reshape to PyTorch style
            X = X.transpose(2, 0, 1).astype(np.float32)
            Z = Z.transpose(2, 0, 1).astype(np.float32)

            return X, Z
        except:
            return None

def collate_fn(batch):
    """Filter out None values from the batch."""
    batch = [x for x in batch if x is not None]
    return torch.utils.data.dataloader.default_collate(batch)
