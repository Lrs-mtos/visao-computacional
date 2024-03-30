import cv2
import numpy as np
import sys

# Load the image
filename = './imagens/jato.jpg'
im = cv2.imread(filename)

# Check if the image was loaded successfully
if im is None:
    print(f"Error: Unable to load image {filename}")
    sys.exit(1)

# Split the image into its individual color channels
b, g, r = cv2.split(im)

# Amplify the red and green channels
g = cv2.addWeighted(g, 1.5, g, 0, 0)
r = cv2.addWeighted(r, 1.5, r, 0, 0)

# Reconstruct the image
im_yellow_tint = cv2.merge((b, g, r))

# Display the image
cv2.imshow('Yellow Tint', im_yellow_tint)
cv2.waitKey(0)
cv2.destroyAllWindows()
