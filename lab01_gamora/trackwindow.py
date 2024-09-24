import cv2
import numpy as np
import sys

def nothing(x):
    pass

cv2.namedWindow('image')
cv2.createTrackbar('Lower Hue', 'image', 0, 180, nothing)
cv2.createTrackbar('Upper Hue', 'image', 0, 180, nothing)

filename = sys.argv[1]
im = cv2.imread(filename)
im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

while(True):
    lh = cv2.getTrackbarPos('Lower Hue', 'image')
    uh = cv2.getTrackbarPos('Upper Hue', 'image')

    lower_val = np.array([lh, 50, 50])
    upper_val = np.array([uh, 255, 255])

    mask = cv2.inRange(im_hsv, lower_val, upper_val)
    res = cv2.bitwise_and(im, im, mask= mask)

    cv2.imshow('image', res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
