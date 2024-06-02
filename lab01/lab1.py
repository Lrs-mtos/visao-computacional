import cv2 
import sys

filename = sys.argv[1]

# Para carregar uma imagem
img = cv2.imread(filename) # Carrega a imagem
im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converte a imagem para escala de cinza
im_dst = cv2.imread(filename, 0) # Carrega a imagem. O 0 indica que a imagem será carregada em escala de cinza

print(im) # Exibe a matriz da imagem
print(im.shape) # Exibe o tamanho da imagem

width = im.shape[1] # Largura da imagem
height = im.shape[0] # Altura da imagem

# Ler todos os pixels da imagem
for y in range(0, height - 1):
    for x in range(0, width - 1):
        px = im.item(y, x) # Retorna o valor do pixel na posição (x, y)
        im_dst.itemset(y, x, 255 - px) # Deixa a imagem negativa por meio da inversão dos valores dos pixels

# Redimensionar a imagem
new_width = int(im.shape[1] * 0.5) # Nova largura
new_height = int(im.shape[0] * 0.5) # Nova altura

dim = (new_width, new_height) # Dimensões da nova imagem
im_resized = cv2.resize(im, dim, interpolation = cv2.INTER_AREA) # Redimensiona a imagem

ret, im_thresh = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY) # Limiariza a imagem

# Para exibir a imagem
cv2.imshow('Imagem', im) # Exibe a imagem
cv2.imshow('Threshold', im_thresh) # Exibe a imagem limiarizada
cv2.imshow('Negativa', im_dst) # Exibe a imagem neagtiva
cv2.waitKey(0) # Espera o usuário pressionar alguma tecla
cv2.destroyAllWindows() # Fecha todas as janelas abertas