# Computer Vision

## Class 12 - 04 OCT 2018

### Efficient Lucas Kanade
- Gradient must be applied on the source image (do not warp before convolution), choose the correct reference frame
- Taylor series expansion always works when pixels are close to each other
- Gauss Newton iteration for efficient computation
- Very sensitive to initialization
- Coarse-to-Fine iteration - Blur a lot in the beginning to get rid of noise for curve fitting and the back off the blurring to not lose the edge features
- Composition of warps, first warp with 0 and add delta P, then warp with P - This makes the initial starting point 0 (the identity warp)
- By **inverse composing** the above, we can warp on the template now and move it over the source image
- LK is susceptible to outliers because of least squares error
- Use robust error functions such as RANSAC, L1 error
- Eigen-objects - Get dominant eigen values from SVD (or PCA)
- Active appearance model is LK with eigen objects
- Descriptor performance - compare 8 neighbors and the comparison output forms a new channel and then run inverse composition