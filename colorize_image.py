import os
import warnings
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from skimage import io, transform
from colorize import network, dataset, util

input_size = (224, 224)

if __name__ == '__main__':
    parser = ArgumentParser(description='Colorize an image.')
    parser.add_argument('model')
    parser.add_argument('out_dir')
    parser.add_argument('images', nargs='+')
    args = parser.parse_args()

    # Load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cnn = network.CNN()
    cnn.load_state_dict(torch.load(args.model, map_location=device))
    cnn.eval()
    cnn.to(device)
    w = torch.tensor(np.load('resources/w.npy')).to(device)

    for i, image in tqdm(enumerate(args.images), total=len(args.images)):
        # Load, normalize, encode and resize
        L, ab_original = util.imread(image, size=None)
        X = (transform.resize(L.copy(), input_size) - 50) / 50
        Z = util.soft_encode(transform.resize(ab_original, input_size))

        # Convert to PyTorch tensor
        X = torch.tensor(X.transpose(2, 0, 1).astype(np.float32)[np.newaxis,...]).to(device)
        Z = torch.tensor(Z.transpose(2, 0, 1).astype(np.float32)[np.newaxis,...]).to(device)

        # Colorize, decode, and resize
        Z_hat = util.reshape(cnn(X).cpu().data.numpy(), 3)[0]
        ab_colorized = transform.resize(util.decode(Z_hat, strategy='annealed_mean'), L.shape[:2])

        # Save
        rgb_original = (util.lab2rgb(L, ab_original) * 255).astype(np.uint8)
        rgb_colorized = (util.lab2rgb(L, ab_colorized) * 255).astype(np.uint8)
        L = (L / 100 * 255).astype(np.uint8)
        io.imsave(os.path.join(args.out_dir, f'{i:04d}_grayscale.jpg'), L)
        io.imsave(os.path.join(args.out_dir, f'{i:04d}_original.jpg'), rgb_original)
        io.imsave(os.path.join(args.out_dir, f'{i:04d}_colorized.jpg'), rgb_colorized)
