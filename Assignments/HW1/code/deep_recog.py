import os
import multiprocessing

import torch
import imageio
import numpy as np
import skimage.transform
import torchvision.transforms

import util
import network_layers

# Standardization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# PyTorch transform objects setup
resize = torchvision.transforms.Resize((224, 224))
normalize = torchvision.transforms.Normalize(mean, std)
image2tensor = torchvision.transforms.ToTensor()
tensor2image = torchvision.transforms.ToPILImage()

def build_recognition_system(vgg16, num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N, K)
    * labels: numpy.ndarray of shape (N)
    '''

    train_data = np.load("../data/train_data.npz")
    n_train = train_data['image_names'].shape[0]
    args = [ (i, train_data['image_names'][i][0], vgg16) for i in range(n_train) ]

    get_image_feature(args[0])
    # get_image_feature((0, ''))

def evaluate_recognition_system(vgg16, num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8, 8)
    * accuracy: accuracy of the evaluated system
    '''

    
    test_data = np.load("../data/test_data.npz")
    # ----- TODO -----
    pass


def preprocess_image(image, toTensor=True):
    '''
    Preprocesses the image to load into the prebuilt network.

    [input]
    * image: numpy.ndarray of shape (H, W, 3)

    [output]
    * image_processed: torch.Tensor of shape (3, H, W)
    '''

    # Convert grayscale images to 3 channels
    if len(image.shape) < 3:
        image = np.dstack([image] * 3)

    # Convert higher channel images to 3 channels
    if image.shape[2] > 3:
        image = image[:, :, :3]

    # Convert to float and normalize
    if toTensor:
        image_processed = normalize(scale(image2tensor(image))).unsqueeze(0)
    else:
        image = image.astype('float')/255
        image = skimage.transform.resize(image, (224, 224, 3))
        image_processed = (image - mean)/std
        
    return image_processed

def get_image_feature(args):
    '''
    Extracts deep features from the prebuilt VGG-16 network.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * image_path: path of image file
    * vgg16: prebuilt VGG-16 network.
    
    [saved]
    * feat: evaluated deep feature
    '''

    i, image_path, vgg16 = args

    # Read image
    image = imageio.imread('../data/' + image_path)
    
    # Preprocess image
    image = preprocess_image(image, toTensor=False)

    # Extract deep features
    feat = network_layers.extract_deep_feature(image, util.get_VGG16_weights())
    # print(feat.shape)

def distance_to_set(feature, train_features):
    '''
    Compute distance between a deep feature with all training image deep features.

    [input]
    * feature: numpy.ndarray of shape (K)
    * train_features: numpy.ndarray of shape (N, K)

    [output]
    * dist: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    
    pass