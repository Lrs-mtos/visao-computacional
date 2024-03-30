
""" Yellow is considered the most luminous 
of the spectrum colors. As noted previously, 
in Red Green Blue (RGB) display space, Yellow 
light is created by adding 'Red' and 'Green' lights 
together. The Hex code for pure Yellow is #FFFF00. 
"""

import cv2
import numpy as np
import sys

from cv_utils import waitKey

def gamma_correction_LUT(img, gamma, c=1.0):
    # cria uma Lookup Table (LUT)
    GAMMA_LUT = np.array([c * ((i / 255.0) ** (1.0 / gamma)) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # aplica a transformação usando LUT
    return cv2.LUT(img, GAMMA_LUT)

def make_yellowish(img, gamma):
    b, g, r = cv2.split(img)
    g = gamma_correction_LUT(g, gamma)
    r = gamma_correction_LUT(r, gamma)
    # Não aplicar a correção ao azul para que a imagem fique amarelada
    return cv2.merge([b, g, r])

def callback_trackbar(x):
    try:
        gamma = cv2.getTrackbarPos('gamma','image')
        yellow_gamma = gamma * 0.01 if gamma > 0 else 1
        im_gamma = make_yellowish(im, yellow_gamma)
        cv2.imshow('image',im_gamma)
    except:
        return
    
#abre a imagem
filename = sys.argv[1]
im = cv2.imread(filename)

if im is None:
    print(f"Error: Unable to load image {filename}")
    sys.exit(1)

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.createTrackbar('gamma', 'image', 100, 200, callback_trackbar)  # Trackbar with gamma values from 1 to 2

# Mostra a imagem modificada
cv2.imshow('image', im)
waitKey('image', 27) #27 = ESC	

cv2.destroyAllWindows()
