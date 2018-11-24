import numpy as np
import skimage.io

from nn import *
from util import *

# params = {}
# initialize_weights(2, 3, params, name='')
# print(params)

# x = np.asarray([[5, 3, 1, -1], [5, 3, 1, -1]])
# print(softmax(x))

def test_softmax_basic():
    """
    Some simple tests to get you started. 
    Warning: these are not exhaustive.
    """
    print("Running basic tests...")
    test1 = softmax(np.array([1, 2]).reshape(1, 2))
    print(test1)
    assert np.amax(np.fabs(test1 - np.array(
        [0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print(test2)
    assert np.amax(np.fabs(test2 - np.array(
        [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))) <= 1e-6

    test3 = softmax(np.array([[-1001,-1002]]).reshape(1, 2))
    print(test3)
    assert np.amax(np.fabs(test3 - np.array(
        [0.73105858, 0.26894142]))) <= 1e-6

# test_softmax_basic()

# avg = 0
# for i in range(1, 5):
#     a = i*2
#     avg = (a + avg*(i-1))/(i)
#     print(a, avg)

# print(avg)

# i = 0
# l = [1, 3, 2, 4, 5, 6, 7, 8]
# while i < len(l):
#     print(l[i])
#     if l[i] == 2:
#         l = l[:i] + [20, 21] + l[i+1:]
#         i += 2
#     i += 1
# print(l)

iters = lambda img: (img.shape[0] * img.shape[1])//int(1.5e6) + 3*int(img.shape[1]>4000)
img = skimage.io.imread('../images/01_list.jpg')
print(iters(img))
img = skimage.io.imread('../images/02_letters.jpg')
print(iters(img))
img = skimage.io.imread('../images/03_haiku.jpg')
print(iters(img))
img = skimage.io.imread('../images/04_deep.jpg')
print(iters(img))
