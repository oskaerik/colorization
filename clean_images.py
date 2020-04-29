import os
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from argparse import ArgumentParser
from skimage import io

def preprocess(path):
    try:
        rgb = io.imread(path)
        if len(rgb.shape) != 3 or rgb.shape[2] != 3:
            return None
        return path
    except:
        return None

if __name__ == '__main__':
    parser = ArgumentParser(description='Preprocess a directory of images and save a file of good images.')
    parser.add_argument('in_dir')
    parser.add_argument('out_file')
    args = parser.parse_args()

    images = glob(os.path.join(args.in_dir, '*.jpg'))

    print(f'Preprocessing {len(images)} images')
    with Pool(16) as pool:
        preprocessed = list(tqdm(pool.imap(preprocess, images), total=len(images)))
    preprocessed = [p for p in preprocessed if p is not None]

    with open(args.out_file, 'w') as f:
        for p in preprocessed:
            print(p, file=f)
    print(f'Saved {len(preprocessed)} images')
