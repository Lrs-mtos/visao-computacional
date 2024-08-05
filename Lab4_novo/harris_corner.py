import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

image_path = sys.argv[1]

# Função para detectar cantos em uma imagem
def detect_corners(image_path):
    # Carrega a imagem

    img = cv2.imread(image_path)

    if img is None:
        print('Não foi possível abrir ou encontrar a imagem')
        return

    # Cópia da imagem original
    image = np.copy(img)

    # Converte a imagem para tons de cinza
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Converte para float32, necessário para o detector de Harris
    gray = np.float32(gray_image)

    #Aplica o detector de cantos Harris
    harris_corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    # Threshold para uma imagem ideal, depende da qualidade da imagem
    image[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]

    # Mostra a imagem original e a imagem com os cantos detectados
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Imagem Original')
    plt.axis('off')


    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Cantos Detectados')
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    detect_corners(image_path)
