import cv2
import numpy as np
import sys

def nothing(x):
    pass

# Create a window
cv2.namedWindow('image')

# Create trackbars for hue, saturation and value
cv2.createTrackbar('Lower Hue', 'image', 0, 180, nothing)
cv2.createTrackbar('Upper Hue', 'image', 0, 180, nothing)
cv2.createTrackbar('Lower Saturation', 'image', 0, 255, nothing)
cv2.createTrackbar('Upper Saturation', 'image', 0, 255, nothing)
cv2.createTrackbar('Lower Value', 'image', 0, 255, nothing)
cv2.createTrackbar('Upper Value', 'image', 0, 255, nothing)

# Load the image
filename = sys.argv[1]
im = cv2.imread(filename)
im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

while True:
    # Get current positions of all trackbars
    lh = cv2.getTrackbarPos('Lower Hue', 'image')
    uh = cv2.getTrackbarPos('Upper Hue', 'image')
    ls = cv2.getTrackbarPos('Lower Saturation', 'image')
    us = cv2.getTrackbarPos('Upper Saturation', 'image')
    lv = cv2.getTrackbarPos('Lower Value', 'image')
    uv = cv2.getTrackbarPos('Upper Value', 'image')

    # Set the lower and upper HSV range according to the values from the trackbars
    lower_val = np.array([lh, ls, lv])
    upper_val = np.array([uh, us, uv])

    # Create a mask and result image
    mask = cv2.inRange(im_hsv, lower_val, upper_val)
    res = cv2.bitwise_and(im, im, mask=mask)

    # Show the images
    cv2.imshow('image', res)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Destroy all windows
cv2.destroyAllWindows()
