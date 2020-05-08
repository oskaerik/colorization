import os
import warnings
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from argparse import ArgumentParser
from skimage import io, transform
from colorize import network, dataset, util

input_size = (224, 224)

if __name__ == '__main__':
    parser = ArgumentParser(description='Copies a directory structure with colorized images.')
    parser.add_argument('model')
    parser.add_argument('in_dir')
    parser.add_argument('out_dir')
    args = parser.parse_args()
    images = glob(os.path.join(args.in_dir, '**', '*.jpg'), recursive=True)
    model_name = os.path.splitext(os.path.split(args.model)[1])[0]
    out_dir = os.path.join(args.out_dir, model_name)

    # Load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cnn = network.CNN()
    cnn.load_state_dict(torch.load(args.model, map_location=device))
    cnn.eval()
    cnn.to(device)
    w = torch.tensor(np.load('resources/w.npy')).to(device)

    for i, image in tqdm(enumerate(images), total=len(images)):
        # Create directory if it does not exits
        path, name = os.path.split(image)
        path = os.path.join(out_dir, path)
        os.makedirs(path, exist_ok=True)

        # Load, normalize, encode and resize
        L, Y = util.imread(image, size=None)
        X = (transform.resize(L.copy(), input_size) - 50) / 50
        Z = util.soft_encode(transform.resize(Y, input_size))

        # Convert to PyTorch tensor
        X = torch.tensor(X.transpose(2, 0, 1).astype(np.float32)[np.newaxis,...]).to(device)
        Z = torch.tensor(Z.transpose(2, 0, 1).astype(np.float32)[np.newaxis,...]).to(device)

        # Colorize, decode, and resize
        Z_hat = util.reshape(cnn(X).cpu().data.numpy(), 3)[0]
        ab = transform.resize(util.decode(Z_hat, strategy='annealed_mean'), L.shape[:2])

        # Save
        rgb = (util.lab2rgb(L, ab) * 255).astype(np.uint8)
        io.imsave(os.path.join(path, name), rgb)
