#!/usr/bin/python3

'''
16-720B Computer Vision (Fall 2018)
Homework 1 - Spatial Pyramid Matching for Scene Classification
'''

__author__ = "16-720B TAs"
__credits__ = ["Simon Lucey", "16-720B TAs"]
__version__ = "1.0.1"

import os

andrew_id = 'hvhavlen'

if ( os.path.isfile('../' + andrew_id + '/' + andrew_id + '/visual_words.py') and \
os.path.isfile('../' + andrew_id + '/' + andrew_id + '/visual_recog.py') and \
os.path.isfile('../' + andrew_id + '/' + andrew_id + '/network_layers.py') and \
os.path.isfile('../' + andrew_id + '/' + andrew_id + '/deep_Recog.py') and \
os.path.isfile('../' + andrew_id + '/' + andrew_id + '/util.py') and \
os.path.isfile('../' + andrew_id + '/' + andrew_id + '/main.py') and \
os.path.isfile('../' + andrew_id + '/' + andrew_id + '_hw1.pdf') ):
    print('file check passed!')
else:
    print('file check failed!')

# modify file name according to final naming policy
# images should be included in the report
