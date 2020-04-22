import sys
import math
import time
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from scipy.special import softmax
from skimage import io, transform, color
from sklearn.neighbors import NearestNeighbors
from . import dataset

bins = np.load('resources/bins.npy')
Q = len(bins)
nn1 = NearestNeighbors(n_neighbors=1)
nn1.fit(bins)
nn5 = NearestNeighbors(n_neighbors=5)
nn5.fit(bins)
input_size = (224, 224)
eps = 1e-16

def closest_bins(Y):
    """Finds the closest bin for each pixel given two color channels Y."""
    Y, _ = reshape(Y, 3)

    # Find nearest neighboring bin for each pixel
    _, indices = nn1.kneighbors(Y.reshape(-1, 2))
    indices = indices.reshape(*Y.shape[:-1], -1)

    return indices

def soft_encode(Y, sigma=5, debug=False):
    """Converts two color channels Y to probability distributiton Z."""
    Y, reshaped = reshape(Y, 3)

    # Find nearest neighboring bins
    distances, indices = nn5.kneighbors(Y.reshape(-1, 2))
    distances = distances.reshape(*Y.shape[:-1], -1)
    indices = indices.reshape(*Y.shape[:-1], -1)

    # Weight proportionally to distance using Gaussian kernel and normalize
    weighted = np.exp(-distances**2 / (2 * sigma**2))
    weighted /= weighted.sum(axis=2, keepdims=True)

    # Build Z matrix
    Z = np.zeros((*Y.shape[:-1], Q))
    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            row = indices[i,j,:]
            Z[i,j,row] = weighted[i,j,:]

    if debug:
        print('Z.shape:', Z.shape)
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                for k in range(indices.shape[2]):
                    assert(Z[i,j,indices[i,j,k]] == weighted[i,j,k])
                    assert(np.isclose(Z[i,j,:].sum(), 1.0))

    if reshaped:
        Z, _ = reshape(Z, 4)

    return Z

def decode(Z, strategy='annealed_mean', T=0.38):
    """Converts probability distribution Z to two color channels Y."""
    Z, reshaped = reshape(Z, 3)

    if strategy == 'annealed_mean':
        Z = softmax(np.log(Z + eps) / T, axis=2)
    elif strategy == 'mode':
        # One-hot to the most probable bin for each pixel
        Z = np.eye(Z.shape[2])[Z.argmax(axis=2)]
    elif strategy == 'expected_value':
        # Do nothing if we want expected value
        pass
    else:
        raise Exception(f'Invalid strategy (use "annealed mean", "mode" or "expected"): {strategy}')

    # Weight each bin proportionally to its probability
    Y = (bins * Z[...,np.newaxis]).sum(axis=2)

    if reshaped:
        Y, _ = reshape(Y, 4)

    return Y

def multinomial_cross_entropy_loss(Z_hat, Z, w):
    """Calculates the weighted multinomial cross entropy loss."""
    q_star = torch.argmax(Z, dim=1, keepdim=True)
    v = torch.take(w, q_star)
    loss = torch.mean(-torch.sum(v * Z * torch.log(Z_hat + eps), dim=(1, 2, 3)))
    return loss

def reshape(a, dims):
    """Reshapes an array between Matplotlib and PyTorch shapes."""
    if len(a.shape) == dims:
        # Do nothing
        return a, False

    if dims == 3:
        # Matplotlib (w, h, c)
        a = np.squeeze(a.transpose(2, 3, 1, 0), axis=3)
    elif dims == 4:
        # PyTorch (1, c, w, h)
        a = a.transpose(2, 0, 1)[np.newaxis,...]
    else:
        raise Exception(f'Invalid dimensions, only 3 or 4 allowed: {dims}')

    return a, True

def imread(path, size=input_size):
    """Reads and resizes an image, returns Lab channels."""
    # Read image as RGB (drop alpha channel if it exists) and resize
    rgb = io.imread(path)[:,:,:3]
    rgb = transform.resize(rgb, size)

    # Convert to Lab and split into L and ab channels
    lab = color.rgb2lab(rgb)
    L = lab[...,:1]
    ab = lab[...,1:]

    return L, ab

def imshow(L, ab):
    """Displays an image, takes Lab channels."""
    L, _ = reshape(L, 3)
    ab, _ = reshape(ab, 3)
    lab = np.concatenate((L, ab), axis=2)

    # Convert back to RGB (while suppressing warnings)
    warnings.simplefilter('ignore')
    rgb = color.lab2rgb(lab)
    warnings.resetwarnings()

    plt.imshow(rgb)
    plt.axis('off')

def side_by_side(*args):
    """Displays two images side by side."""
    fig = plt.figure(figsize=(len(args)*4, len(args)*2))
    for i, arg in enumerate(args, start=1):
        L, ab, title = arg
        ax = fig.add_subplot(1, len(args), i)
        imshow(L, ab)
        ax.set_title(title)
    plt.show()

def train(net, paths, device, epochs, log=True, batch_size=32, shuffle=True, num_workers=8):
    """Trains a network on a set of images."""

    def logger(string):
        """Prints to stdout and log file."""
        string = ''.join(['[', str(datetime.now()), '] ', str(string)])
        print(string)
        print(string, file=log_file, flush=True)

    log_file = open(f'logs/log_{int(time.time() * 1000)}.txt', 'w')
    logger(f'Opened log file: {log_file.name}')

    data = dataset.Dataset(paths)
    dataloader = torch.utils.data.DataLoader(data,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers)
    w = torch.tensor(np.load('resources/w.npy')).to(device)

    net.train()
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=3*10**-5, betas=(0.9, 0.99), weight_decay=10**-3)

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch, (X, Z) in tqdm(enumerate(dataloader, start=1), total=math.ceil(len(data) / batch_size)):
            X, Z = X.to(device), Z.to(device)
            optimizer.zero_grad()

            Z_hat = net(X)
            loss = multinomial_cross_entropy_loss(Z_hat, Z, w)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        logger(f'Epoch: {epoch}/{epochs}, Loss: {running_loss / batch}')
        torch.save(net.state_dict(), f'models/model_{epoch}.pth')

    log_file.close()

def colorize_images(net, paths, device):
    """Trains a network on a set of images."""
    data = dataset.Dataset(paths)
    dataloader = torch.utils.data.DataLoader(data)
    w = torch.tensor(np.load('resources/w.npy')).to(device)

    net.eval()
    net.to(device)

    for X, Z in dataloader:
        # Colorize
        X, Z = X.to(device), Z.to(device)
        Z_hat = net(X)

        # Unnormalize, decode, transform and show images
        L = X.cpu().data.numpy() * 50 + 50
        Z, Z_hat = reshape(Z.cpu().data.numpy(), 3)[0], reshape(Z_hat.cpu().data.numpy(), 3)[0]
        ab_gt = transform.resize(decode(Z, strategy='mode'), input_size)
        ab_annealed = transform.resize(decode(Z_hat, strategy='annealed_mean'), input_size)
        ab_mode = transform.resize(decode(Z_hat, strategy='mode'), input_size)
        ab_expected = transform.resize(decode(Z_hat, strategy='expected_value'), input_size)
        side_by_side((L, ab_gt, 'Ground truth'),
                     (L, ab_annealed, 'Annealed-mean'),
                     (L, ab_mode, 'Mode'),
                     (L, ab_expected, 'Expected value'))
