# Computer Vision

## Class 4 - 06 SEP 2018

### Edge Detection Filters (Canny)
- Edges need to be oriented
- Edges need to be scalable to detect both sudden and gradual changes
- Slide 7, lecture 4 (Canny) - Image with strongest response of all the output filter image responses - this new image is single channel, also store the index of the channel where the response was maximum
- Here, **argmax** is the non-linearity applied to a bank of filter responses, in CNNs, we use ReLU
- Steerability - Linear combination of horizontal and vertical filters to produce all orientations - Used to optimize (Slide 8)
- Canny quantized to four directions (0, 30, 60, 90) - Chosen based on cross validation (Slide 3)
- Problem with choosing filter for all angles
	- Noise can affect the bin in which response will fall
	- The max function can be very sensitive to noise and hence affect reconstruction (??)
- CNNs force some filters to be steerable to increase speed (??)
- Non-maximal suppression, sparse.. (??)
- High threshold for dominant lines and low threshold for continuation for the line (this is called hysteresis), chosen empirically

### Texture
- High enough K value in K means clustering to avoid noisy responses

### TODO
- [ ] Implement Canny edge detector in Python
- [ ] Revise K means and slides
