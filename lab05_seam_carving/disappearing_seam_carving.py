import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, color

drawing = False  # True if the mouse is pressed
ix, iy = -1, -1

def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(img, (x, y), 15, (0, 0, 0), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img, (x, y), 15, (0, 0, 0), -1)

def calculate_energy(image):
    gray_image = color.rgb2gray(image)
    energy = np.abs(filters.sobel_h(gray_image)) + np.abs(filters.sobel_v(gray_image))
    return energy

def find_seam_vertical(energy):
    r, c = energy.shape
    M = energy.copy()
    backtrack = np.zeros_like(M, dtype=int)

    for i in range(1, r):
        for j in range(c):
            if j == 0:
                idx = np.argmin(M[i-1, j:j+2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i-1, j-1:j+2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i-1, idx + j - 1]
            M[i, j] += min_energy

    return M, backtrack

def remove_seam(image, backtrack):
    r, c, _ = image.shape
    output = np.zeros((r, c - 1, 3), dtype=image.dtype)
    j = np.argmin(backtrack[-1])
    for i in reversed(range(r)):
        output[i, :, 0] = np.delete(image[i, :, 0], [j])
        output[i, :, 1] = np.delete(image[i, :, 1], [j])
        output[i, :, 2] = np.delete(image[i, :, 2], [j])
        j = backtrack[i, j]
    return output

def seam_carving(image, num_seams, energy):
    for _ in range(num_seams):
        M, backtrack = find_seam_vertical(energy)
        image = remove_seam(image, backtrack)
        energy = np.delete(energy, np.argmin(backtrack[-1]), axis=1)
    return image

# Load image
img = io.imread('/home/larissa/Desktop/UFC/Visão/Labs/images/balls.jpg')

# Allow the user to draw
cv2.namedWindow('Draw to remove')
cv2.setMouseCallback('Draw to remove', draw_circle)

while True:
    cv2.imshow('Draw to remove', img)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

cv2.destroyAllWindows()

# Convert the drawn area to grayscale and create an energy map
mask = color.rgb2gray(img) == 0  # The drawn area will be black (zero energy)
energy = calculate_energy(img)
energy[mask] = 0  # Set the energy of the drawn area to zero

# Apply seam carving
new_image = seam_carving(img, 160, energy)

# Display the result
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax[0].imshow(io.imread('/home/larissa/Desktop/UFC/Visão/Labs/images/balls.jpg'))
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(new_image)
ax[1].set_title('Seam Carved Image')
ax[1].axis('off')

plt.tight_layout()
plt.show()
