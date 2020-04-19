"""Genererates the weighting factors w."""
import glob
import numpy as np
import scipy.ndimage as ndi
from tqdm import tqdm
from colorize import util

if __name__ == '__main__':
    images = glob.glob("data/**/*.jpg")

    # Find empirical distribution
    p = np.zeros(len(util.bins))
    for path in tqdm(images):
        _, Y = util.imread(path)
        closest_bins = util.closest_bins(Y).flatten()
        for closest_bin in closest_bins:
            p[closest_bin] += 1
    p /= p.sum()

    # Smoothen with Gaussian filter
    p_tilde = ndi.gaussian_filter(p, sigma=5)

    # Mix with uniform distribution and normalize
    w = 1/(p_tilde + 1/len(p_tilde))
    w /= (p_tilde*w).sum()

    np.save('w.npy', w)
