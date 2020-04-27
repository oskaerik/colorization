import numpy as np
from skimage.io import imread
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.animation as animation

DOG_IMAGES_FOLER = 'dog_images'

def index_str(index):
    """Given the frame index of a dog video, get the corresponding index string for the .jpg image"""
    index_str = str(index)
    return '0' * (8 - len(index_str)) + index_str

def output_video(img_array, filename):
    """Given an array of images with size (f, w, h, 3) outputs animated video filename.mp4"""
    fig = plt.figure()
    imgs = [[plt.imshow(img, animated=True)] for img in img_array]
    ani = animation.ArtistAnimation(fig, imgs, interval=33.3)
    ani.save(f'{filename}.mp4')

def load_videos(ranges):
    """Given an array of shot ranges with size (shot, start, end), load all the videos into an array"""
    return np.array([[imread(DOG_IMAGES_FOLER + index_str(index) + '.jpg')  for index in range(start, end + 1)] for shot, start, end in ranges])

    


