import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from skimage import filters
from skimage import color

def calculate_energy(image):
    # Converte a imagem para escala de cinza
    gray_image = color.rgb2gray(image)
    # Calcula o gradiente da imagem
    energy = filters.sobel(gray_image)
    return np.abs(energy)

# Funções para seam carving vertical
def find_vertical_seam(energy):
    r, c = energy.shape
    M = energy.copy()
    backtrack = np.zeros_like(M, dtype=int)

    # Preenchendo a matriz de energia acumulada de cima para baixo
    for i in range(1, r):
        for j in range(c):
            # Trata as bordas separadamente
            if j == 0:
                idx = np.argmin(M[i-1, j:j+2])
                min_energy = M[i-1, idx + j]
                backtrack[i, j] = idx + j
            elif j == c - 1:
                idx = np.argmin(M[i-1, j-1:j+1])
                min_energy = M[i-1, idx + j - 1]
                backtrack[i, j] = idx + j - 1
            else:
                idx = np.argmin(M[i-1, j-1:j+2])
                min_energy = M[i-1, idx + j - 1]
                backtrack[i, j] = idx + j - 1
            M[i, j] += min_energy

    return M, backtrack

def remove_vertical_seam(image, M, backtrack, seam_mask):
    r, c, _ = image.shape
    output = np.zeros((r, c - 1, 3), dtype=image.dtype)
    j = np.argmin(M[-1])  # Inicia a partir do pixel com menor energia acumulada na última linha
    for i in reversed(range(r)):
        seam_mask[i, j] = True  # Marca o pixel da costura a ser removida
        output[i] = np.delete(image[i], j, axis=0)
        j = backtrack[i, j]
    return output, seam_mask

# Funções para seam carving horizontal
def find_horizontal_seam(energy):
    r, c = energy.shape
    M = energy.copy()
    backtrack = np.zeros_like(M, dtype=int)

    # Preenchendo a matriz de energia acumulada da esquerda para a direita
    for j in range(1, c):
        for i in range(r):
            # Trata as bordas separadamente
            if i == 0:
                idx = np.argmin(M[i:i+2, j-1])
                min_energy = M[idx + i, j-1]
                backtrack[i, j] = idx + i
            elif i == r - 1:
                idx = np.argmin(M[i-1:i+1, j-1])
                min_energy = M[idx + i - 1, j-1]
                backtrack[i, j] = idx + i - 1
            else:
                idx = np.argmin(M[i-1:i+2, j-1])
                min_energy = M[idx + i - 1, j-1]
                backtrack[i, j] = idx + i - 1
            M[i, j] += min_energy

    return M, backtrack

def remove_horizontal_seam(image, M, backtrack, seam_mask):
    r, c, _ = image.shape
    output = np.zeros((r - 1, c, 3), dtype=image.dtype)
    i = np.argmin(M[:, -1])  # Inicia a partir do pixel com menor energia acumulada na última coluna
    for j in reversed(range(c)):
        seam_mask[i, j] = True  # Marca o pixel da costura a ser removida
        output[:, j] = np.delete(image[:, j], i, axis=0)
        i = backtrack[i, j]
    return output, seam_mask

# Função principal que combina ambos
def seam_carving(image, num_seams_vertical=0, num_seams_horizontal=0):
    img = image.copy()
    seam_mask_vertical = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
    seam_mask_horizontal = np.zeros((img.shape[0], img.shape[1]), dtype=bool)

    # Remoção de costuras verticais
    for _ in range(num_seams_vertical):
        energy = calculate_energy(img)
        M, backtrack = find_vertical_seam(energy)
        img, seam_mask_vertical = remove_vertical_seam(img, M, backtrack, seam_mask_vertical)

    # Remoção de costuras horizontais
    for _ in range(num_seams_horizontal):
        energy = calculate_energy(img)
        M, backtrack = find_horizontal_seam(energy)
        img, seam_mask_horizontal = remove_horizontal_seam(img, M, backtrack, seam_mask_horizontal)

    # Combina as máscaras de costura
    # Ajusta o tamanho das máscaras se necessário
    if seam_mask_vertical.shape != seam_mask_horizontal.shape:
        # Redimensiona a máscara menor para coincidir com a maior
        max_r = max(seam_mask_vertical.shape[0], seam_mask_horizontal.shape[0])
        max_c = max(seam_mask_vertical.shape[1], seam_mask_horizontal.shape[1])
        seam_mask_vertical_resized = np.zeros((max_r, max_c), dtype=bool)
        seam_mask_horizontal_resized = np.zeros((max_r, max_c), dtype=bool)
        seam_mask_vertical_resized[:seam_mask_vertical.shape[0], :seam_mask_vertical.shape[1]] = seam_mask_vertical
        seam_mask_horizontal_resized[:seam_mask_horizontal.shape[0], :seam_mask_horizontal.shape[1]] = seam_mask_horizontal
        seam_mask = seam_mask_vertical_resized | seam_mask_horizontal_resized
    else:
        seam_mask = seam_mask_vertical | seam_mask_horizontal

    return img, seam_mask

# Código principal
if __name__ == "__main__":
    # Carregar a imagem
    img = io.imread('/home/larissa/Desktop/UFC/Visão/Labs/images/flor.jpg')  # Substitua pelo caminho da sua imagem

    # Reduzir a resolução da imagem (opcional)
    img_resized = transform.resize(img, (img.shape[0] // 2, img.shape[1] // 2), anti_aliasing=True)

    # Definir o número de costuras a serem removidas
    num_seams_vertical = 10    # Número de costuras verticais
    num_seams_horizontal = 40  # Número de costuras horizontais

    # Aplicar o seam carving
    new_image, seam_mask = seam_carving(img_resized, num_seams_vertical, num_seams_horizontal)

    # Redimensionar a máscara de costura para o tamanho original da imagem
    seam_mask_resized = transform.resize(seam_mask, (img.shape[0], img.shape[1]), anti_aliasing=False, preserve_range=True).astype(bool)

    # Desenhar linhas vermelhas nas costuras removidas
    img_with_seams = img.copy()
    img_with_seams[seam_mask_resized] = [255, 0, 0]  # Desenha as costuras em vermelho

    # Mostrar a imagem original, com costuras e a modificada
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    ax[0].imshow(img)
    ax[0].set_title('Imagem Original')
    ax[0].axis('off')

    ax[1].imshow(img_with_seams)
    ax[1].set_title('Localização das Costuras')
    ax[1].axis('off')

    ax[2].imshow(new_image)
    ax[2].set_title('Imagem com Seam Carving')
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()
