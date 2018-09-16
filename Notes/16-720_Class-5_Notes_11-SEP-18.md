# Computer Vision

## Class 5 - 11 SEP 2018

### Bag of Words
- Create number of patches and compare histogram of different images
- Replace words with textons and patches with oriented filters
- Textons are repeating patterns, measure using statistics
- Bases on K-means cluster indices of dictionary
- Bank of filters include Laplacians and oriented edges
- X Matrix (slide 6)
	- M - Number of filters
	- N - Number of images/pixels in the training set
- The active cell refers to the dictionary and the column in Z refers to column in X
 
### Neural Networks
- ReLU makes the matrix sparse
- Pooling acts as a LPF, so less features need to be learned
- Trade off between learning features vs keep spatial invariance 
