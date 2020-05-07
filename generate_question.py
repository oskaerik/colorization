import os
import warnings
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser
from skimage import io, transform
from colorize import network, dataset, util

input_size = (224, 224)

if __name__ == '__main__':
    parser = ArgumentParser(description='Generate a question for the form.')
    parser.add_argument('coco')
    parser.add_argument('dogs')
    parser.add_argument('out_dir')
    parser.add_argument('images', nargs='+')
    args = parser.parse_args()

    # Load models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    coco = network.CNN()
    coco.load_state_dict(torch.load(args.coco, map_location=device))
    coco.eval()
    coco.to(device)
    dogs = network.CNN()
    dogs.load_state_dict(torch.load(args.dogs, map_location=device))
    dogs.eval()
    dogs.to(device)
    w = torch.tensor(np.load('resources/w.npy')).to(device)

    # Set a/b order, if order=0 then coco has index 0, and if order=1 then coco has index 1
    order = np.random.randint(2, size=len(args.images))
    np.save(os.path.join(args.out_dir, 'order.npy'), order)
    with open(os.path.join(args.out_dir, 'images.pkl'), 'wb') as f:
        pickle.dump(args.images, f)
    letters = ['a', 'b']

    for i, image in tqdm(enumerate(args.images), total=len(args.images)):
        # Load, normalize, encode and resize
        L, ab_original = util.imread(image, size=None)
        X = (transform.resize(L.copy(), input_size) - 50) / 50
        Z = util.soft_encode(transform.resize(ab_original, input_size))

        # Convert to PyTorch tensor
        X = torch.tensor(X.transpose(2, 0, 1).astype(np.float32)[np.newaxis,...]).to(device)
        Z = torch.tensor(Z.transpose(2, 0, 1).astype(np.float32)[np.newaxis,...]).to(device)

        # Colorize, decode, and resize
        Z_hat_coco = util.reshape(coco(X).cpu().data.numpy(), 3)[0]
        Z_hat_dogs = util.reshape(dogs(X).cpu().data.numpy(), 3)[0]
        ab_coco = transform.resize(util.decode(Z_hat_coco, strategy='annealed_mean'), L.shape[:2])
        ab_dogs = transform.resize(util.decode(Z_hat_dogs, strategy='annealed_mean'), L.shape[:2])

        # Shuffle
        rgb_coco = (util.lab2rgb(L, ab_coco) * 255).astype(np.uint8)
        rgb_dogs = (util.lab2rgb(L, ab_dogs) * 255).astype(np.uint8)
        rgb = np.stack((rgb_coco, rgb_dogs))

        fig = plt.figure(figsize=[s*2//25 for s in L.shape[:2]])
        for j in range(0, 2):
            ax = fig.add_subplot(1, 2, j+1)
            plt.imshow(rgb[(order[i] + j) % 2])
            plt.axis('off')
            ax.set_title(letters[j], fontsize=36)
        fig.savefig(os.path.join(args.out_dir, f'{i:04d}_question.png'), bbox_inches='tight')
        plt.close()
