import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform, color
from sklearn.neighbors import NearestNeighbors

bins = np.load('resources/bins.npy')
Q = len(bins)
nn1 = NearestNeighbors(n_neighbors=1)
nn1.fit(bins)
nn5 = NearestNeighbors(n_neighbors=5)
nn5.fit(bins)

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

def decode(Z):
    """Converts probability distribution Z to two color channels Y."""
    Z, reshaped = reshape(Z, 3)

    # TODO: Implement annealed-mean
    mode = Z.argmax(axis=2)
    Y = np.zeros((*mode.shape, 2))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j,:] = bins[mode[i,j]]

    if reshaped:
        Y, _ = reshape(Y, 4)

    return Y

def multinomial_cross_entropy_loss(Z_hat, Z, eps=1e-16):
    """Calculates the weighted multinomial cross entropy loss."""
    # TODO: Implement weighting
    loss = torch.mean(torch.sum(-Z * torch.log(Z_hat + eps), dim=(1, 2, 3)))
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

def imread(path, size=(224, 224)):
    """Reads and resizes an image, returns Lab channels."""
    rgb = io.imread(path)
    rgb = transform.resize(rgb, size)
    lab = color.rgb2lab(rgb)
    L = lab[...,:1]
    ab = lab[...,1:]
    return L, ab

def imshow(L, ab):
    """Displays an image, takes Lab channels."""
    L, _ = reshape(L, 3)
    ab, _ = reshape(ab, 3)
    lab = np.concatenate((L, ab), axis=2)
    rgb = color.lab2rgb(lab)
    plt.figure()
    plt.imshow(rgb)
    plt.show()
