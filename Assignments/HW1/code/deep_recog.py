import os
import multiprocessing

import scipy
import torch
import imageio
import numpy as np
import skimage.transform
import torchvision.transforms

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
NPROC = 2 # util.get_num_CPU()

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

    # Load training data
    train_data = np.load("../data/train_data.npz")
    n_train = train_data['image_names'].shape[0]
    labels = np.asarray(train_data['labels'])

    # Multiprocess feature extraction
    # TODO: Remove NPROC
    args = [ (i, train_data['image_names'][i][0], vgg16) for i in range(n_train) ]
    pool = multiprocessing.Pool(processes=NPROC)
    features = np.asarray(pool.imap(get_image_feature, args))
    pool.close()
    pool.join()

    # Save all the trained data
    np.savez_compressed('trained_system_deep', features=features, labels=labels)

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
    # TODO: Remove NPROC
    args = [ (test_data['image_names'][i][0], features, train_labels, vgg16) for i in range(n_test) ]
    pool = multiprocessing.Pool(processes=NPROC)
    predicted_labels = np.asarray(pool.imap(predict_image, args))
    pool.close()
    pool.join()

    np.save('predicted_labels_deep', predicted_labels)

    # Evaluate the metrics
    confusion_matrix = sklearn.metrics.confusion_matrix(test_labels, predicted_labels)
    accuracy_score = sklearn.metrics.accuracy_score(test_labels, predicted_labels)

    print('Confusion Matrix:\n', confusion_matrix)
    print('Accuracy Score:', accuracy_score)

    return confusion_matrix, accuracy_score

def evaluate_custom_implementation(vgg16):
    '''
    TODO: Document this function
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
    image = skimage.transform.resize(image, (224, 224, 3))
    image_processed = (image - mean)/std

    # Convert to PyTorch tensor
    if USE_PYTORCH: image_processed = image2tensor(image_processed).unsqueeze(0)

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

    return feat

def distance_to_set(feature, train_features):
    '''
    Compute distance between a deep feature with all training image deep features.

    [input]
    * feature: numpy.ndarray of shape (K)
    * train_features: numpy.ndarray of shape (N, K)

    [output]
    * dist: numpy.ndarray of shape (N)
    '''

    # Calculate euclidean distances for every feature
    dist = np.asarray(scipy.spatial.distance.cdist(train_features, feature))
    return dist

def predict_image(args):
    '''
    TODO: Document this function
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
