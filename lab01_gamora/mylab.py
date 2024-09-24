import sys
import cv2
import numpy as np

filename = sys.argv[1]
im = cv2.imread(filename)

if im is None:
    print(f"Error: Unable to load image {filename}")
    sys.exit(1)

im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

lower_blue = np.array([94, 47, 0])
upper_blue = np.array([118, 255, 255])
lower_green = np.array([7, 50, 50])
upper_green = np.array([75, 255, 255])

center_x = im_hsv.shape[1] // 2

right_side = im_hsv[:, center_x:] 

left_side = im_hsv[:, :center_x]

mask_green_right = cv2.inRange(right_side, lower_green, upper_green) 
mask_blue_left = cv2.inRange(left_side, lower_blue, upper_blue) 

right_side[mask_green_right > 0, 0] = 118
left_side[mask_blue_left > 0, 0] = 75 

im_hsv[:, center_x:] = right_side

im_hsv[:, :center_x] = left_side

im_bgr = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)

cv2.imshow('Modified Image', im_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()