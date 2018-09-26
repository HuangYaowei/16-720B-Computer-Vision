#!/usr/bin/python3

'''
16-720B Computer Vision (Fall 2018)
Homework 1 - Spatial Pyramid Matching for Scene Classification
'''

__author__ = "Heethesh Vhavle"
__credits__ = ["Simon Lucey", "16-720B TAs"]
__version__ = "1.0.1"
__email__ = "heethesh@cmu.edu"

# External modules
import imageio
import numpy as np
import torchvision

# Local python modules
import util
import visual_words
import visual_recog
import deep_recog

if __name__ == '__main__':

    # Get CPU cores
    num_cores = util.get_num_CPU()

    # Load image
    path_img = "../data/kitchen/sun_aasmevtpkslccptd.jpg"
    image = imageio.imread(path_img)
    image = image.astype('float')/255

    # Extract filter responses
    filter_responses = visual_words.extract_filter_responses(image)
    util.display_filter_responses(filter_responses)

    # Dictionary and wordmap generation
    visual_words.compute_dictionary(num_workers=num_cores)
    dictionary = np.load('dictionary.npy')
    wordmap = visual_words.get_visual_words(image, dictionary)

    # Build and evaluate SPM recognition system
    visual_recog.build_recognition_system(num_workers=num_cores)
    conf, accuracy = visual_recog.evaluate_recognition_system(num_workers=num_cores)
    print(conf)
    print(np.diag(conf).sum()/conf.sum())

    # Load VGG-16 pretrained model
    vgg16 = torchvision.models.vgg16(pretrained=True).double()
    vgg16.eval()

    # Build and evaluate SPM recognition system
    deep_recog.build_recognition_system(vgg16, num_workers=num_cores//2)
    conf, accuracy = deep_recog.evaluate_recognition_system(vgg16, num_workers=num_cores//2)
    print(conf)
    print(np.diag(conf).sum()/conf.sum())

    # Verify the custom VGG-16 implementation results
    deep_recog.evaluate_custom_implementation(vgg16)
