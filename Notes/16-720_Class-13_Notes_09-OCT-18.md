# Computer Vision

## Class 13 - 09 OCT 2018

### Circulant Toeplitz Matrix
- If X is a CTM, in SVD of X, U and V are the Fourier Transform
```
U, S, V.T = svd(X)
F, S, F.T = svd(X)
```
- The F matrix is dense, sparse is good for fast computation
- F is compositionally sparse - can be formed by bunch of sparse matrices