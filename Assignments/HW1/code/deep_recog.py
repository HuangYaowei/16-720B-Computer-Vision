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
import multiprocessing

# External modules
import scipy
import torch
import imageio
import numpy as np
import skimage.transform
import torchvision.transforms

# Local python modules
import util
import network_layers

# PyTorch enable
USE_PYTORCH = True

# PyTorch globals
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
image2tensor = torchvision.transforms.ToTensor()

# Multiprocess globals
PROGRESS = 0
PROGRESS_LOCK = multiprocessing.Lock()
NPROC = util.get_num_CPU()//2
TEMP_PATH = '../data/deep_features'

def build_recognition_system(vgg16, num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [saved]
    * trained_system.npz: numpy compressed file with following contents
        - features: numpy.ndarray of shape (N, K)
        - labels: numpy.ndarray of shape (N)
    '''

    # Load training data
    train_data = np.load("../data/train_data.npz")
    n_train = train_data['image_names'].shape[0]
    labels = np.asarray(train_data['labels'])

    # Create folders to save features
    if not os.path.exists(TEMP_PATH):
        os.makedirs(TEMP_PATH)

    # Multiprocess feature extraction
    args = [ (i, train_data['image_names'][i][0], vgg16) for i in range(n_train) ]
    pool = multiprocessing.Pool(processes=num_workers)
    pool.map(get_image_feature, args)
    pool.close()
    pool.join()

    # Load the save extracted features
    features = [ np.load(os.path.join(TEMP_PATH, '%d.npy'%i)) for i in range(n_train) ]

    # Save all the trained data
    np.savez_compressed('trained_system_deep', features=features, labels=labels)

def evaluate_recognition_system(vgg16, num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8, 8)
    * accuracy: accuracy of the evaluated system
    '''
    
    global PROGRESS
    PROGRESS = 0

    # Testing data
    test_data = np.load("../data/test_data.npz")
    test_labels = test_data['labels']
    n_test = test_data['image_names'].shape[0]
    file_paths = [ test_data['image_names'][i][0] for i in range(n_test) ]

    # Trained system data
    trained_system = np.load("trained_system_deep.npz")    
    features = trained_system['features']
    train_labels = trained_system['labels']

    # Multiprocess feature extraction
    args = [ (test_data['image_names'][i][0], features, train_labels, vgg16) for i in range(n_test) ]
    pool = multiprocessing.Pool(processes=num_workers)
    predicted_labels = np.asarray(pool.map(predict_image, args))
    pool.close()
    pool.join()

    # Evaluate the metrics
    confusion_matrix, accuracy_score = util.confusion_matrix_and_accuracy(test_labels, predicted_labels)

    print('Confusion Matrix:\n', confusion_matrix)
    print('Accuracy Score:', accuracy_score)

    return confusion_matrix, accuracy_score

def evaluate_custom_implementation(vgg16):
    '''
    Evaluates the custom implementation with PyTorch outputs

    [input]
    * vgg16: prebuilt VGG-16 network.

    [output]
    * comparison: boolean representing success/failure
    '''

    global USE_PYTORCH
    
    # Load training data
    train_data = np.load("../data/train_data.npz")
    n_train = train_data['image_names'].shape[0]

    # Pick a random image from the dataset
    args = (0, train_data['image_names'][0][0], vgg16) # np.random.choice(n_train)

    # Extract features using custom implementation
    USE_PYTORCH = True 
    custom_features = get_image_feature(args)

    # Extract features using PyTorch implementation
    USE_PYTORCH = False
    torch_features = get_image_feature(args)

    comparison = np.allclose(custom_features, torch_features)
    print('\nComparison Result > Implementations Match:', comparison)

    return comparison

def preprocess_image(image):
    '''
    Preprocesses the image to load into the prebuilt network

    [input]
    * image: numpy.ndarray of shape (H, W, 3)

    [output]
    * image_processed: torch.Tensor of shape (1, 3, H, W) (USE_PYTORCH=True)
    * image_processed: numpy.ndarray of shape (H, W, 3) (USE_PYTORCH=False)
    '''

    # Convert grayscale images to 3 channels
    if len(image.shape) < 3:
        image = np.dstack([image] * 3)

    # Convert higher channel images to 3 channels
    if image.shape[2] > 3:
        image = image[:, :, :3]

    # Convert to float and normalize
    image = skimage.transform.resize(image, (224, 224, 3))
    image_processed = (image - mean)/std

    # Convert to PyTorch tensor
    if USE_PYTORCH: image_processed = image2tensor(image_processed).unsqueeze(0)

    return image_processed

def get_image_feature(args):
    '''
    Extracts deep features from the prebuilt VGG-16 network
    This is a function run by a subprocess

    [input]
    * i: index of training image
    * image_path: path of image file
    * vgg16: prebuilt VGG-16 network
    
    [saved]
    * feat: evaluated deep feature
    '''

    global PROGRESS
    with PROGRESS_LOCK: PROGRESS += NPROC
    
    i, image_path, vgg16 = args
    print('Processing: %04d/1440 | Index: %04d | Image: %s'%(PROGRESS, i, image_path))

    # Read and preprocess image
    image = imageio.imread('../data/' + image_path)    
    image = preprocess_image(image)

    # Extract deep features
    if USE_PYTORCH:
        feat = vgg16.features(image).flatten() # VGG-16 Features
        feat = vgg16.classifier[:5](feat).detach().numpy().flatten() # VGG-16 Classifiers
    else:
        feat = network_layers.extract_deep_feature(image, util.get_VGG16_weights())

    # Save the extracted features
    np.save('%s/%d'%(TEMP_PATH, i), feat)
    return feat

def distance_to_set(feature, train_features):
    '''
    Compute distance between a deep feature with all training image deep features

    [input]
    * feature: numpy.ndarray of shape (K)
    * train_features: numpy.ndarray of shape (N, K)

    [output]
    * dist: numpy.ndarray of shape (N)
    '''

    # Calculate euclidean distances for every feature
    dist = np.asarray(scipy.spatial.distance.cdist(train_features, np.expand_dims(feature, axis=0)))
    return dist

def predict_image(args):
    '''
    Predicts the label using the trained system and extracted VGG-16 features
    This is a function run by a subprocess.

    [input]
    * image_path: path of image file
    * features: trained features from VGG-16
    * train_labels: trained set of labels 
    * vgg16: prebuilt VGG-16 network
    
    [output]
    * predicted_label: int representing the predicted label
    '''

    global PROGRESS
    with PROGRESS_LOCK: PROGRESS += NPROC

    image_path, features, train_labels, vgg16 = args
    print('Processing: %03d/160 | Image: %s'%(PROGRESS, image_path))

    # Read and preprocess image
    image = imageio.imread('../data/' + image_path)
    image = preprocess_image(image)

    # Extract deep features
    if USE_PYTORCH:
        feat = vgg16.features(image).flatten() # VGG-16 Features
        feat = vgg16.classifier[:5](feat).detach().numpy().flatten() # VGG-16 Classifiers
    else:
        feat = network_layers.extract_deep_feature(image, util.get_VGG16_weights())

    # Find the predicted label
    predicted_label = train_labels[np.argmin(distance_to_set(feat, features))]
    return predicted_label
