import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, color

drawing = False  # Verdadeiro se o mouse está pressionado
ix, iy = -1, -1

def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, img_mask

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(img, (x, y), 15, (0, 0, 0), -1)
            cv2.circle(img_mask, (y, x), 15, 255, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img, (x, y), 15, (0, 0, 0), -1)
        cv2.circle(img_mask, (y, x), 15, 255, -1)

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
                min_energy = M[i-1, idx + j]
                backtrack[i, j] = idx + j
            elif j == c -1:
                idx = np.argmin(M[i-1, j-1:j+1])
                min_energy = M[i-1, idx + j -1]
                backtrack[i, j] = idx + j -1
            else:
                idx = np.argmin(M[i-1, j-1:j+2])
                min_energy = M[i-1, idx + j -1]
                backtrack[i, j] = idx + j -1
            M[i, j] += min_energy

    return M, backtrack

def remove_seam(image, M, backtrack):
    r, c, _ = image.shape
    output = np.zeros((r, c - 1, 3), dtype=image.dtype)
    j = np.argmin(M[-1])  # Linha corrigida
    seam_idx = []
    for i in reversed(range(r)):
        output[i, :, :] = np.delete(image[i, :, :], j, axis=0)
        seam_idx.append(j)
        j = backtrack[i, j]
    seam_idx.reverse()
    return output, seam_idx

def remove_seam_energy(energy, seam_idx):
    r, c = energy.shape
    output = np.zeros((r, c -1), dtype=energy.dtype)
    for i in range(r):
        output[i, :] = np.delete(energy[i, :], seam_idx[i])
    return output

def remove_seam_mask(mask, seam_idx):
    r, c = mask.shape
    output = np.zeros((r, c -1), dtype=mask.dtype)
    for i in range(r):
        output[i, :] = np.delete(mask[i, :], seam_idx[i])
    return output

def seam_carving(image, num_seams, mask):
    img = image.copy()
    energy_mask = mask.copy()
    for _ in range(num_seams):
        energy = calculate_energy(img)
        # Definir energia negativa nas áreas selecionadas
        energy[energy_mask > 0] = -1000  # Valor negativo grande
        M, backtrack = find_seam_vertical(energy)
        img, seam_idx = remove_seam(img, M, backtrack)
        energy_mask = remove_seam_mask(energy_mask, seam_idx)
    return img

# Carregar a imagem
img = io.imread('/home/larissa/Desktop/UFC/Visão/Labs/images/balls.jpg')

# Criar uma máscara para as áreas desenhadas
img_mask = np.zeros(img.shape[:2], dtype=np.uint8)

# Permitir que o usuário desenhe
cv2.namedWindow('Desenhe para remover')
cv2.setMouseCallback('Desenhe para remover', draw_circle)

while True:
    cv2.imshow('Desenhe para remover', img)
    if cv2.waitKey(1) & 0xFF == 27:  # Pressione 'ESC' para sair
        break

cv2.destroyAllWindows()

# Aplicar o seam carving
num_seams = 160  # Ajuste conforme necessário
new_image = seam_carving(img, num_seams, img_mask)

# Exibir o resultado
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax[0].imshow(io.imread('/home/larissa/Desktop/UFC/Visão/Labs/images/balls.jpg'))
ax[0].set_title('Imagem Original')
ax[0].axis('off')

ax[1].imshow(new_image)
ax[1].set_title('Imagem após Seam Carving')
ax[1].axis('off')

plt.tight_layout()
plt.show()
