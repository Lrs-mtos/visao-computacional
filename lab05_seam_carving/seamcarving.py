import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from skimage import filters
from skimage import color

def calculate_energy(image):
    # Convertendo a imagem para escala de cinza
    gray_image = color.rgb2gray(image)
    # Calculando o gradiente da imagem
    energy = np.abs(filters.sobel_h(gray_image))
    return energy

def find_seam(energy):
    # A energia acumulada ao longo do caminho mínimo
    r, c = energy.shape
    M = energy.copy()
    backtrack = np.zeros_like(M, dtype=int) 
    #print(backtrack)

    # Preenchendo a matriz de energia acumulada
    for i in range(1, r):
        for j in range(c):
            # Bordas são tratadas separadamente
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

def remove_seam(image, M, backtrack, seam_mask):
    r, c, _ = image.shape
    output = np.zeros((r, c - 1, 3), dtype=image.dtype)
    j = np.argmin(M[-1])
    #j = np.argmin(backtrack[-1])
    for i in reversed(range(r)):
        seam_mask[i, j] = True  # Marca o pixel da costura a ser removida
        output[i] = np.delete(image[i], j, axis=0)
        j = backtrack[i, j]
    return output, seam_mask

def seam_carving(image, num_seams):
    r, c, _ = image.shape
    seam_mask = np.zeros((r, c), dtype=bool)  # Máscara para armazenar as costuras removidas
    for _ in range(num_seams):
        energy = calculate_energy(image)
        M, backtrack = find_seam(energy)
        image, seam_mask = remove_seam(image, M, backtrack, seam_mask)
    return image, seam_mask

# Carregar a imagem
#img = io.imread('/home/larissa/Desktop/UFC/Visão/Labs/images/flor.jpg')
img = io.imread('balls.jpg')

# Reduzir a resolução da imagem
img_resized = transform.resize(img, (img.shape[0] // 2, img.shape[1] // 2), anti_aliasing=True)

# Aplicar o seam carving
new_image, seam_mask = seam_carving(img_resized, 40)  # Reduz 40 costuras verticais

# Redimensionar a máscara de costura para o tamanho original da imagem
seam_mask_resized = transform.resize(seam_mask, (img.shape[0], img.shape[1]), anti_aliasing=False, preserve_range=True).astype(bool)

# Desenhar linhas vermelhas nas costuras removidas
img_with_seams = img.copy()
img_with_seams[seam_mask_resized] = [255, 0, 0]  # Desenha as costuras em vermelho

# Mostrar a imagem original, com costuras e a modificada
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(img_with_seams)
ax[1].set_title('Seam Locations')
ax[1].axis('off')

ax[2].imshow(new_image)
ax[2].set_title('Seam Carved Image')
ax[2].axis('off')

plt.tight_layout()
plt.show()
