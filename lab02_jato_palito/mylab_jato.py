
""" Yellow is considered the most luminous 
of the spectrum colors. As noted previously, 
in Red Green Blue (RGB) display space, Yellow 
light is created by adding 'Red' and 'Green' lights 
together. The Hex code for pure Yellow is #FFFF00. 
https://uxdesign.cc/when-red-and-green-become-yellow-79187f9fa6ec#:~:text=As%20noted%20previously%2C%20in%20Red,for%20pure%20Yellow%20is%20%23FFFF00.
"""

import cv2
import numpy as np
import sys

from cv_utils import waitKey

def gamma_correction_LUT(img, gamma, c=1.0):
    # Create a Lookup Table (LUT)
    GAMMA_LUT = np.array([c * ((i / 255.0) ** (1.0 / gamma)) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # Apply the transformation using LUT
    return cv2.LUT(img, GAMMA_LUT)

def make_yellowish(img, gamma):
    b, g, r = cv2.split(img)
    g = gamma_correction_LUT(g, gamma)
    r = gamma_correction_LUT(r, gamma)
    # Do not apply the correction to blue to make the image more yellowish
    return cv2.merge([b, g, r])

def callback_trackbar(x):
    try:
        # Invert the gamma value, so 100 -> gamma=1, and 0 -> gamma=2
        gamma = cv2.getTrackbarPos('gamma', 'image')
        # Adjust gamma scale
        yellow_gamma = (gamma * 0.01) + 1  # Makes gamma range between 1 and 2
        im_gamma = make_yellowish(im, yellow_gamma)
        cv2.imshow('image', im_gamma)
    except:
        return

filename = sys.argv[1]
im = cv2.imread(filename)

if im is None:
    print(f"Error: Unable to load image {filename}")
    sys.exit(1)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.createTrackbar('gamma', 'image', 0, 100, callback_trackbar)

cv2.imshow('image', im)
#waitKey('image', 27)  # 27 = ESC
cv2.waitKey(0)
cv2.destroyAllWindows()
