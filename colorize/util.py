import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

bins = np.load('resources/bins.npy')
Q = len(bins)
neighbors = NearestNeighbors(n_neighbors=5)
neighbors.fit(bins)

def soft_encode(Y, sigma=5, debug=False):
    """Converts two color channels Y to probability distributiton Z."""
    if len(Y.shape) == 4:
        Y = np.squeeze(Y.transpose(2, 3, 1, 0), axis=3)
        reshaped = True
    else:
        reshaped = False

    # Find nearest neighboring bins
    distances, indices = neighbors.kneighbors(Y.reshape(-1, 2))
    distances = distances.reshape(*Y.shape[:-1], -1)
    indices = indices.reshape(*Y.shape[:-1], -1)

    # Weight proportionally to distance using Gaussian kernel and normalize
    weighted = np.exp(-distances**2 / (2 * sigma**2))
    weighted /= weighted.sum(axis=2)[...,np.newaxis]

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
        Z = Z.transpose(2, 0, 1)[np.newaxis,...]
    return Z

def decode(Z):
    if len(Z.shape) == 4:
        Z = np.squeeze(Z.transpose(2, 3, 1, 0), axis=3)
        reshaped = True
    else:
        reshaped = False

    mode = Z.argmax(axis=2)
    Y = np.zeros((*mode.shape, 2))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j,:] = bins[mode[i,j]]

    if reshaped:
        Y = Y.transpose(2, 0, 1)[np.newaxis,...]
    return Y

def multinomial_cross_entropy_loss(Z_hat, Z, eps=1e-16):
    """Calculates the weighted multinomial cross entropy loss."""
    # TODO: Implement weighting
    loss = torch.mean(-Z * torch.log(Z_hat + eps))
    return loss
