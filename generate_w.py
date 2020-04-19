"""Genererates the weighting factors w."""
import glob
import numpy as np
import scipy.ndimage as ndi
from multiprocessing import Pool
from tqdm import tqdm
from colorize import util

def bins_from_image(path):
    """Reads and image and counts the occurences of each bin."""
    _, Y = util.imread(path)
    closest_bins = util.closest_bins(Y).flatten()
    count = np.zeros(len(util.bins))
    for closest_bin in closest_bins:
        count[closest_bin] += 1
    return count

if __name__ == '__main__':
    images = glob.glob("data/**/*.jpg")

    # Find empirical distribution
    with Pool(16) as p:
        p = np.sum(list(tqdm(p.imap(bins_from_image, images), total=len(images))), axis=0)
    p /= p.sum()

    # Smoothen with Gaussian filter
    p_tilde = ndi.gaussian_filter(p, sigma=5)

    # Mix with uniform distribution and normalize
    w = 1/(p_tilde + 1/len(p_tilde))
    w /= (p_tilde*w).sum()

    np.save('resources/w.npy', w)
