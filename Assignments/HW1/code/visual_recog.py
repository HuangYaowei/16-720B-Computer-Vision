import os
import math
import time
import queue
import multiprocessing

import imageio
import numpy as np

import util
import visual_words

# Globals
PROGRESS = 0
PROGRESS_LOCK = multiprocessing.Lock()

def build_recognition_system(num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N, M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K, 3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    # Load data
    train_data = np.load("../data/train_data.npz")
    dictionary = np.load("dictionary.npy")

    # Hyperparameters
    layer_num = 3
    K = dictionary.shape[0]
    n_train = train_data['image_names'].shape[0]

    # Training images file path list
    file_paths = [ train_data['image_names'][i][0] for i in range(n_train) ]

    # Multiprocess feature extraction using SPM
    pool = multiprocessing.Pool(processes=num_workers)
    pool.map(get_image_feature, file_paths, dictionary, layer_num, K)
    pool.close()
    pool.join()

def evaluate_recognition_system(num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8, 8)
    * accuracy: accuracy of the evaluated system
    '''

    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("trained_system.npz")

    # TODO

def get_image_feature(file_path, dictionary, layer_num, K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K*(4^layer_num-1)/3))
    '''

    global PROGRESS
    with PROGRESS_LOCK: PROGRESS += 8
    print('Processing: %04d/1440 | Index: %04d | Image: %s'%(PROGRESS, i, file_path))

    # Read image
    image = imageio.imread('../data/' + file_path)
    image = image.astype('float')/255

    # Create visual wordmap for the image
    wordmap = visual_words.get_visual_words(image, dictionary)

    # Compute histogram features of visual words using spatial pyramid matching
    feature = get_feature_from_wordmap_SPM(wordmap, layer_num, K)
    return feature

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K*(4^layer_num-1)/3))
    * histograms: numpy.ndarray of shape (T, K*(4^layer_num-1)/3))

    [output]
    * sim: numpy.ndarray of shape (T)
    '''
    
    # Compute histogram similarity
    sim = np.asarray([ np.sum(np.minimum(word_hist, hist)) for hist in histograms ])
    return sim

def split_image(array, levels):
    '''
    TODO: Document this
    '''

    # Level 0 returns original array
    splits = [array]

    # Level > 1 splits the array into 2^(levels) blocks
    for level in range(1, levels + 1):
        splits.append(np.asarray([np.array_split(row, 2**level, 1) for row in np.array_split(array, 2**level, 0)]))
    
    # Index of this list contains the level at which the cells are divided
    return np.asarray(splits)

def compute_histogram(array, bins, normalize_pixels=None, disp=False):
    '''
    TODO: Document
    [input]
    * wordmap: numpy.ndarray of shape (H, W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    # Compute histogram
    hist, bin_edges = np.histogram(array, bins=bins)

    # L1 normalize
    if normalize_pixels: hist = hist / normalize_pixels

    # Display histogram
    if disp: util.display_histogram(hist, bins=bins)

    return hist

def get_feature_from_wordmap(wordmap, dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H, W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    
    # Compute histogram
    hist = np.asarray(compute_histogram(wordmap, dict_size))
    return hist

def get_feature_from_wordmap_SPM(wordmap, layer_num, dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H, W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''

    # Parameters
    levels = layer_num - 1
    total_pixels = wordmap.shape[0] * wordmap.shape[1]
    
    # Generate weights for the histogram layers
    weights = [2**(-levels)]*2 + [ 2**(l - levels - 1) for l in range(2, levels + 1) ]
    
    # Split the image into list of layers of 2^(level) cells
    layers = split_image(wordmap, levels)

    # Form all layers hists
    hist_all = [compute_histogram(layers[0], dict_size, total_pixels) * weights[0]]
    for level, weight in zip(layers[1:], weights[1:]):
        for cell_row in level:
            for cell in cell_row:
                hist_all.append(compute_histogram(cell, dict_size, total_pixels) * weight)

    # Concatenate all histogram features
    hist_all = np.concatenate(hist_all)
    return hist_all
