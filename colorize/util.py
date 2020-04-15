import numpy as np
from sklearn.neighbors import NearestNeighbors

bins = np.load('resources/bins.npy')
Q = len(bins)
neighbors = NearestNeighbors(n_neighbors=5)
neighbors.fit(bins)

def soft_encoding(Y, sigma=5, debug=False):
    """Converts two color channels Y to probability distributiton Z."""
    Y_shape = Y.shape

    # Find nearest neighboring bins
    distances, indices = neighbors.kneighbors(Y.reshape(-1, 2))
    distances = distances.reshape(*Y_shape[:-1], -1)
    indices = indices.reshape(*Y_shape[:-1], -1)

    # Weight proportionally to distance using Gaussian kernel and normalize
    weighted = np.exp(-distances**2 / (2 * sigma**2))
    weighted /= weighted.sum(axis=2)[...,np.newaxis]

    # Build Z matrix
    Z = np.zeros((*Y_shape[:-1], Q))
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

    return Z
