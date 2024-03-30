import sys
import cv2
import numpy as np

# Load the image
filename = sys.argv[1]
im = cv2.imread(filename)

# Check if the image was loaded successfully
if im is None:
    print(f"Error: Unable to load image {filename}")
    sys.exit(1)

# Convert BGR to HSV
im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

# Define the range for blue and green colors
lower_blue = np.array([94, 47, 0])
upper_blue = np.array([118, 255, 255])
lower_green = np.array([7, 50, 50])
upper_green = np.array([75, 255, 255])

# Find the center of the image to split it into left and right
center_x = im_hsv.shape[1] // 2

# Work only on the right side of the image for green color changes
right_side = im_hsv[:, center_x:]

# Now get the left side of the image
left_side = im_hsv[:, :center_x]

# Create masks for blue and green colors on the right side
mask_green_right = cv2.inRange(right_side, lower_green, upper_green)
mask_blue_left = cv2.inRange(left_side, lower_blue, upper_blue)

# Change green to blue (Hue for blue is around 120) on the right side
right_side[mask_green_right > 0, 0] = 120

# Change blue to green (Hue for green is around 60) on the left side
left_side[mask_blue_left > 0, 0] = 44

# Put the modified right side back
im_hsv[:, center_x:] = right_side

# Put the modified left side back
im_hsv[:, :center_x] = left_side

# Convert back to BGR
im_bgr = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)

# Display the image
cv2.imshow('Modified Image', im_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()