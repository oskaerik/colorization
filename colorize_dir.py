import os
import math
import warnings
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from argparse import ArgumentParser
from skimage import io, transform
from colorize import network, util

class Dataset(torch.utils.data.Dataset):
    def __init__(self, paths, in_size=(224, 224), out_size=(56, 56)):
        self.paths = paths
        self.in_size = in_size
        self.out_size = out_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        L, Y = util.imread(path, size=None)
        X = (transform.resize(L.copy(), self.in_size) - 50) / 50
        X = X.transpose(2, 0, 1).astype(np.float32)
        return X, L, path

def collate_fn(batch):
    Xs, Ls, paths = zip(*batch)
    Xs = torch.utils.data.dataloader.default_collate(Xs)
    return Xs, Ls, paths

if __name__ == '__main__':
    parser = ArgumentParser(description='Copies a directory structure with colorized images.')
    parser.add_argument('model')
    parser.add_argument('in_dir')
    parser.add_argument('out_dir')
    args = parser.parse_args()
    images = []
    for ext in ['jpg', 'JPEG']:
        images.extend(glob(os.path.join(args.in_dir, '**', f'*.{ext}'), recursive=True))
    model_name = os.path.splitext(os.path.split(args.model)[1])[0]
    out_dir = os.path.join(args.out_dir, model_name)
    for image in images:
        # Create directory if it does not exits
        path = os.path.split(image)[0]
        path = os.path.join(out_dir, path)
        os.makedirs(path, exist_ok=True)

    # Load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cnn = network.CNN()
    cnn.load_state_dict(torch.load(args.model, map_location=device))
    cnn.eval()
    cnn.to(device)
    w = torch.tensor(np.load('resources/w.npy')).to(device)

    # Create DataLoader
    batch_size = 32
    data = Dataset(images)
    dataloader = torch.utils.data.DataLoader(data,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=8,
                                            collate_fn=collate_fn)

    for batch, (Xs, Ls, paths) in tqdm(enumerate(dataloader), total=math.ceil(len(data) / batch_size)):
        Xs = Xs.to(device)
        Z_hats = cnn(Xs).cpu().data.numpy()
        for Z_hat, L, path in zip(Z_hats, Ls, paths):
            Z_hat = Z_hat.transpose(1, 2, 0)
            ab = transform.resize(util.decode(Z_hat, strategy='annealed_mean'), L.shape[:2])
            rgb = (util.lab2rgb(L, ab) * 255).astype(np.uint8)
            io.imsave(os.path.join(out_dir, path), rgb)
