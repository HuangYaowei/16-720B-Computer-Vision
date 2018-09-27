#!/usr/bin/python3

'''
16-720B Computer Vision (Fall 2018)
Homework 1 - Spatial Pyramid Matching for Scene Classification
'''

__author__ = "Heethesh Vhavle"
__credits__ = ["Simon Lucey", "16-720B TAs"]
__version__ = "1.0.1"
__email__ = "heethesh@cmu.edu"

# In-built modules
import os
import math
import multiprocessing

# External modules
import imageio
import skimage
import numpy as np
import scipy.ndimage
import sklearn.cluster
import scipy.spatial.distance
import matplotlib.pyplot as plt

# Local python modules
import util

# Globals
PROGRESS = 0
PROGRESS_LOCK = multiprocessing.Lock()
NPROC = util.get_num_CPU()
SAMPLED_RESPONSES_PATH = '../data/sampled_responses'

def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''

    # Gaussian SD (sigma) values
    sigmas = [1, 2, 4, 8, 8*math.sqrt(2)]

    # Convert grayscale images to 3 channels
    if len(image.shape) < 3:
        image = np.dstack([image] * 3)

    # Convert higher channel images to 3 channels
    if image.shape[2] > 3:
        image = image[:, :, :3]
    
    # Convert RGB to LAB color space
    image_lab = skimage.color.rgb2lab(image)

    filter_responses = []
    for sigma in sigmas:
        # Gaussian filter
        for i in range(0, 3): 
            filter_responses.append(scipy.ndimage.gaussian_filter(image_lab[:, :, i], sigma=sigma))
        
        # Laplacian of gaussian filter
        for i in range(0, 3): 
            filter_responses.append(scipy.ndimage.gaussian_laplace(image_lab[:, :, i], sigma=sigma))
        
        # X axis derivative of gaussian + Y axis gaussian filter
        for i in range(0, 3): 
            filter_responses.append(scipy.ndimage.gaussian_filter(image_lab[:, :, i], sigma=sigma, order=[0, 1]))
        
        # Y axis derivative of gaussian + X axis gaussian filter
        for i in range(0, 3): 
            filter_responses.append(scipy.ndimage.gaussian_filter(image_lab[:, :, i], sigma=sigma, order=[1, 0]))

    # Stack all 4 filters * 3 channels * 5 sigmas (60) channels
    filter_responses = np.dstack(filter_responses)
    return filter_responses

def get_visual_words(image, dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''

    # Extract filter responses and reshape to (H*W, 3F)
    filter_responses = extract_filter_responses(image)
    filter_responses = filter_responses.reshape(filter_responses.shape[0] * filter_responses.shape[1], filter_responses.shape[2])

    # Calculate euclidean distances (H*W, K) for each pixel with dictionary 
    euclidean_distances = scipy.spatial.distance.cdist(filter_responses, dictionary)

    # Create a wordmap with every pixel equal to the k index with least distance and reshape to (H, W)
    wordmap = np.asarray([ np.argmin(pixel) for pixel in euclidean_distances ]).reshape(image.shape[0], image.shape[1])

    # Display wordmap
    # util.display_image(wordmap, cmap='gist_ncar')
    
    return wordmap

def compute_dictionary_one_image(args):
    '''
    Extracts random samples of the dictionary entries from an image
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file
    * time_start: time stamp of start time

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha, 3F)
    '''

    global PROGRESS
    with PROGRESS_LOCK: PROGRESS += NPROC

    i, alpha, image_path = args
    print('Processing: %04d/1440 | Index: %04d | Image: %s'%(PROGRESS, i, image_path))

    # Read image
    image = imageio.imread('../data/' + image_path)
    image = image.astype('float')/255
    
    # Extract filter responses
    filter_responses = extract_filter_responses(image)

    # Randomly select alpha responses for all channels
    sampled_response = []
    for j in range(alpha):
        ran_h = np.random.choice(image.shape[0])
        ran_w = np.random.choice(image.shape[1])
        sampled_response.append(filter_responses[ran_h, ran_w, :])
    
    # Save the sampled responses
    np.save('%s/%d'%(SAMPLED_RESPONSES_PATH, i), np.asarray(sampled_response))

def compute_dictionary(num_workers=2):
    '''
    Creates the dictionary of visual words by clustering using k-means

    [input]
    * num_workers: number of workers to process in parallel
    
    [saved]
    * dictionary: numpy.ndarray of shape (K, 3F)
    '''

    # Load train images data
    train_data = np.load('../data/train_data.npz')

    # Create folders to save filter responses
    if not os.path.exists(SAMPLED_RESPONSES_PATH):
        os.makedirs(SAMPLED_RESPONSES_PATH)

    # Hyperparameters
    alpha = 150
    n_clusters = 100
    n_train = train_data['image_names'].shape[0]

    # Multiprocess feature extraction and sampling
    args = [ (i, alpha, train_data['image_names'][i][0]) for i in range(n_train) ]
    pool = multiprocessing.Pool(processes=num_workers)
    pool.map(compute_dictionary_one_image, args)
    pool.close()
    pool.join()

    # Load all saved sample responses
    sampled_responses = [ np.load(os.path.join(SAMPLED_RESPONSES_PATH, sample_path)) for sample_path in os.listdir(SAMPLED_RESPONSES_PATH) ]
    sampled_responses = np.asarray(sampled_responses).reshape(alpha * n_train, 60)

    # Cluster using K-means and save the dictionary
    print('Clustering into K-means...')
    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, n_jobs=num_workers).fit(sampled_responses)
    dictionary = kmeans.cluster_centers_
    np.save('dictionary', dictionary)
