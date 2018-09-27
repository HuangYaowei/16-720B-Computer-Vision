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

def check_file(file):
    if os.path.isfile(file):
        return True
    else:
        print('{} not found!'.format(file))
        return False
    
if ( check_file('../'+andrew_id+'/code/visual_words.py') and \
     check_file('../'+andrew_id+'/code/visual_recog.py') and \
     check_file('../'+andrew_id+'/code/network_layers.py') and \
     check_file('../'+andrew_id+'/code/deep_recog.py') and \
     check_file('../'+andrew_id+'/code/util.py') and \
     check_file('../'+andrew_id+'/code/main.py') and \
     check_file('../'+andrew_id+'/code/trained_system.npz') and \
     check_file('../'+andrew_id+'/code/trained_system_deep.npz') and \
     check_file('../'+andrew_id+'/code/dictionary.npy') and \
     check_file('../'+andrew_id+'/'+andrew_id+'_hw1.pdf') ):
    print('file check passed!')
else:
    print('file check failed!')

# modify file name according to final naming policy
# images should be included in the report
