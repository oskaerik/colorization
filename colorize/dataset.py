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
        # Load image
        X, Y = util.imread(self.paths[index])

        # Normalize X between -1 and 1
        X = (X - 50) / 50

        # Resize and encode
        Y = transform.resize(Y, self.out_size)
        Z = util.soft_encode(Y)

        # Reshape to PyTorch style
        X = X.transpose(2, 0, 1).astype('float32')
        Z = Z.transpose(2, 0, 1).astype('float32')

        return X, Z
