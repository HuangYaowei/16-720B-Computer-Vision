# Computer Vision

## Class 11 - 02 OCT 2018

### Efficient Descriptors
- FAST is a binary **interest point detector** to approximate Laplacian
- The comparison (1 or 0) is what makes it fast
- BRIEF is a binary **descriptor**
- BRIEF naturally offers photometry invariance
- Use Hamming distance instead of Euclidean for binary
- ORB combines FAST and BRIEF, and hence 2 orders faster than SIFT
- SIFT is better than ORB in accuracy
- RANSAC throws away the outliers

### Lucas Kanade Algorithm
- Involves linearizing the registration using Taylor series expansion