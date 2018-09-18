# Computer Vision

## Class 7 - 18 SEP 2018

### Pin Hole Cameras
- These images are flipped (UD/LR) due to rays crossing each other
- Optical center is always at the origin
- x/f = u/w = tan(theta-x)
- y/f = v/w = tan(theta-y)
- Focal length affects the FOV, inversely 
- Aperture controls the blur 
- Use bias formula to offset centers
- Intrinsic - Rotation, Bias, Skew and Focal Lengths
- Extrinsic - How camera reference frame with respect to the world

### Transformations
- Rotation matrices do not form a convex set
- Making any problem in CS convex makes it easier to solve
- Lie groups and quaternions are used to make rotations convex
- Camera calibration is calibrating both intrinsic and non-intrinsic parameters
