import sys
import cv2
import numpy as np
import cv_utils as waitKey
import matplotlib.pyplot as plt

# Carrega as imagens
circle_img = cv2.imread('./imagens/circle.jpg')
line_img = cv2.imread('./imagens/line.jpg')

# Checa se as imagens foram carregadas corretamente
if circle_img is None or line_img is None:
    print(f"Error: Unable to load image")
    sys.exit(1)

# Pega a dimensão das imagens
width_circle  = circle_img.shape[1]
height_circle = circle_img.shape[0]

height_line = line_img.shape[0]
width_line = line_img.shape[1]

#threshold
ret,circle_img = cv2.threshold(circle_img,200,255,cv2.THRESH_BINARY)
ret,line_img = cv2.threshold(line_img,200,255,cv2.THRESH_BINARY)

dim_final = (300, 300)

# Escalas
scale_head = np.float32([[1, 0, 0], [0, 1, 0]])  # Escala do círculo. Mantendo o tamanho original
scale_body = np.float32([[1, 0, 0], [0, 1, 0]])  # Escala da linha. Mantendo o tamanho original
scale_arm = np.float32([[0.75, 0, 0], [0, 1, 0]])  # Escala do braço. 75% do tamanho do tronco. Largura do braço foi mantida, o comprimento do braço contém 75% do comprimento do tronco
scale_leg = np.float32([[1.5, 0, 0], [0, 1.5, 0]])  # Escala da perna. Dobro do tamanho dos braços

# Cabeça
head = cv2.bitwise_not(circle_img) # Invertendo as cores da imagem
head = cv2.warpAffine(head, scale_head, dim_final)  # Não modifica a imagem, apenas redimensiona para 300x300
M_translation_c = np.float32([[1, 0, 100], [0, 1, 10]]) # Posicionando a cabeça no centro da imagem, X=100, Y=10.
# A imagem do circulo é um quadrado 100x100. Para centralizar a imagem dentro de um quadrado 300x300, é necessário deslocar 100 pixels para a direita.

im_translated_c = cv2.warpAffine(head, M_translation_c, dim_final) # Deslocando a imagem para a posição correta
head = cv2.bitwise_not(im_translated_c) # Invertendo as cores da imagem

body = cv2.bitwise_not(line_img) # Invertendo as cores da imagem
body = cv2.warpAffine(body, scale_body, dim_final) # Não modifica a imagem, apenas redimensiona para 300x300
#rotate
x_center_line = width_line/2
y_center_line = height_line/2
M_rotation_l = cv2.getRotationMatrix2D((x_center_line,y_center_line),90,1)
body = cv2.warpAffine(body, M_rotation_l, dim_final) # Rotacionando a linha
M_translation_l = np.float32([[1, 0, 100], [0, 1, 72]]) # Posicinando o tronco abaixo da cabeça, X=100, Y=72
body = cv2.warpAffine(body, M_translation_l, dim_final) # Deslocando a imagem para a posição correta
body = cv2.bitwise_not(body) # Invertendo as cores da imagem

# Braço esquerdo - 75% do tamanho do tronco
left_arm = cv2.bitwise_not(line_img) # Invertendo as cores da imagem
left_arm = cv2.warpAffine(left_arm, scale_arm, dim_final) # Redimensionando a imagem para 300x300

M_translation_arm_left = np.float32([[1, 0, 79], [0, 1, 65]]) # Posicinando o braço esquerdo abaixo da cabeça, X=79, Y=65
left_arm = cv2.warpAffine(left_arm, M_translation_arm_left, dim_final) # Deslocando a imagem para a posição correta
left_arm = cv2.bitwise_not(left_arm) # Invertendo as cores da imagem

# Braço direito - 75% do tamanho do tronco
right_arm = cv2.bitwise_not(line_img) # Invertendo as cores da imagem 
right_arm = cv2.warpAffine(right_arm, scale_arm, dim_final) # Redimensionando a imagem para 300x300

M_translation_arm_right = np.float32([[1, 0, 145], [0, 1, 65]]) # Posicinando o braço direito abaixo da cabeça, X=145, Y=65
right_arm = cv2.warpAffine(right_arm, M_translation_arm_right, dim_final) # Deslocando a imagem para a posição correta
right_arm = cv2.bitwise_not(right_arm) # Invertendo as cores da imagem


# Perna esquerda - Dobro do tamanho dos braços
left_leg = cv2.bitwise_not(line_img) # Invertendo as cores da imagem
left_leg = cv2.warpAffine(left_leg, scale_leg, dim_final) # Redimensionando a imagem para 300x300
M_rotation_leg_left = cv2.getRotationMatrix2D((width_line / 2, height_line / 2), 45, 1) # Rotacionando a linha em 45 graus
left_leg = cv2.warpAffine(left_leg, M_rotation_leg_left, dim_final) # Rotacionando a linha
M_translation_leg_left = np.float32([[1, 0, 22], [0, 1, 156]]) # Posicinando a perna esquerda abaixo do tronco, X=22, Y=156
left_leg = cv2.warpAffine(left_leg, M_translation_leg_left, dim_final) # Deslocando a imagem para a posição correta
left_leg = cv2.bitwise_not(left_leg)


# Perna direita - Dobro do tamanho dos braços
right_leg = cv2.bitwise_not(line_img) # Invertendo as cores da imagem
right_leg = cv2.warpAffine(right_leg, scale_leg, dim_final) # Redimensionando a imagem para 300x300
M_rotation_leg_right = cv2.getRotationMatrix2D((width_line / 2, height_line / 2), -45, 1) # Rotacionando a linha em -45 graus
right_leg = cv2.warpAffine(right_leg, M_rotation_leg_right, dim_final) # Rotacionando a linha
M_translation_leg_right = np.float32([[1, 0, 142], [0, 1, 122]]) # Posicinando a perna direita abaixo do tronco, X=142, Y=122
right_leg = cv2.warpAffine(right_leg, M_translation_leg_right, dim_final) # Deslocando a imagem para a posição correta
right_leg = cv2.bitwise_not(right_leg)


img = cv2.bitwise_and(head, body) # Cabeça e tronco
img = cv2.bitwise_and(img, left_arm) # Cabeça, tronco e braço esquerdo
img = cv2.bitwise_and(img, right_arm) # Cabeça, tronco, braço esquerdo e braço direito
img = cv2.bitwise_and(img, left_leg) # Cabeça, tronco, braço esquerdo, braço direito e perna esquerda
img = cv2.bitwise_and(img, right_leg) # Cabeça, tronco, braço esquerdo, braço direito, perna esquerda e perna direita

""" plt.subplot(321), plt.imshow(head, cmap='gray')
plt.subplot(322), plt.imshow(body, cmap='gray')
plt.subplot(323), plt.imshow(left_arm, cmap='gray')
plt.subplot(324), plt.imshow(right_arm, cmap='gray')
plt.subplot(325), plt.imshow(left_leg, cmap='gray')
plt.subplot(326), plt.imshow(right_leg, cmap='gray') """

plt.figure()
plt.title("Resultado Final")
plt.imshow(img)
plt.show()
waitKey('resultado', 27)
