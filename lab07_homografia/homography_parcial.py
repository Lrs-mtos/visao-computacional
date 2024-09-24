import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

image1 = cv2.imread("Imagens/campus_quixada1.png")
image2 = cv2.imread("Imagens/campus_quixada2.png")


#reduz o tamanho das imagens para melhor visualização
h1 = image1.shape[0]
w1 = image1.shape[1]
image1 = cv2.resize(image1, (int(w1*0.5), int(h1*0.5)))

h2 = image2.shape[0]
w2 = image2.shape[1]
image2 = cv2.resize(image2, (int(w2*0.5), int(h2*0.5)) )

img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# find the keypoints and descriptors with SIFT
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)


bf = cv2.BFMatcher()

matches = bf.knnMatch(des1, des2, k=2)
# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])


if len(good) >= 4:

    # Extrair localizações dos bons matches
    pts1 = []
    pts2 = []
    for m in good:
        pts1.append(kp1[m[0].queryIdx].pt)
        pts2.append(kp2[m[0].trainIdx].pt)

    # matrix points
    points1 = np.float32(pts1).reshape(-1, 1, 2)
    points2 = np.float32(pts2).reshape(-1, 1, 2)

    # Encontrar homografia usando RANSAC
    transformation_matrix, inliers = cv2.findHomography(points1, points2, cv2.RANSAC)

    print(transformation_matrix)
    print(inliers)
else:
    raise AssertionError("No enough keypoints.")




##################################
# Obter dimensões das duas imagens
# height1, width1 = img1.shape[:2]
# height2, width2 = img2.shape[:2]
# # Calcula a transformação de toda a imagem1
# img1_transformed = cv2.warpPerspective(image1, transformation_matrix, (width2, height2))
#
# # Criar uma imagem de saída grande o suficiente para segurar ambas as imagens
# height = max(height1, height2)
# width = width1 + width2
# output_image = np.zeros((height, width, 3), dtype=np.uint8)
#
# # Posicionar a imagem transformada na imagem de saída
# output_image[0:height2, 0:width2] = img1_transformed
#
# # Posicionar img2 na primeira metade da imagem de saída
# output_image[0:height2, 0:width2] = image2
#
#
#
# # Exibir a imagem combinada
# cv2.imshow("Imagem Combinada", output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#############################333
height, width = img2.shape
img1_transformed = cv2.warpPerspective(image1, transformation_matrix, (width, height))

# Criar uma imagem para mostrar a combinação
combined_img = cv2.addWeighted(img1_transformed, 0.5, image2, 0.5, 0)


# Mostrar a imagem combinada
cv2.imshow("Imagem Combinada", combined_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


################################################3
height1, width1 = img1.shape[:2]
height2, width2 = img2.shape[:2]


img1_transformed = cv2.warpPerspective(image1, transformation_matrix, (width1+width2, height1+height2))
img1_transformed[0:image2.shape[0], 0:image2.shape[1]] = image2


cv2.imshow("result", img1_transformed)
cv2.imshow("image1", image1)
cv2.imshow("image2", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()



