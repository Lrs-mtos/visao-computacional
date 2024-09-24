import cv2
import numpy as np

# Carregar as imagens
image1 = cv2.imread("Imagens/campus_quixada1.png")
image2 = cv2.imread("Imagens/campus_quixada2.png")

# Reduzir o tamanho das imagens para melhor visualização
h1, w1 = image1.shape[:2]
h2, w2 = image2.shape[:2]
image1 = cv2.resize(image1, (int(w1 * 0.5), int(h1 * 0.5)))
image2 = cv2.resize(image2, (int(w2 * 0.5), int(h2 * 0.5)))

# Converter para escala de cinza
img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Encontrar keypoints e descritores com SIFT
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Correspondência de descritores
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Aplicar teste de razão
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

    # Converter para numpy arrays
    points1 = np.float32(pts1).reshape(-1, 1, 2)
    points2 = np.float32(pts2).reshape(-1, 1, 2)

    # Encontrar homografia usando RANSAC
    transformation_matrix, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Obter dimensões das imagens
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # Definir os cantos das imagens
    corners_image1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners_image2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    # Transformar os cantos da imagem1
    transformed_corners_image1 = cv2.perspectiveTransform(corners_image1, transformation_matrix)

    # Combinar os cantos para encontrar o tamanho da imagem resultante
    all_corners = np.concatenate((transformed_corners_image1, corners_image2), axis=0)

    # Encontrar os limites da nova imagem
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Calcular a translação necessária
    translation_dist = [-x_min, -y_min]

    # Criar matriz de translação
    H_translation = np.array([[1, 0, translation_dist[0]],
                              [0, 1, translation_dist[1]],
                              [0, 0, 1]])

    # Aplicar a transformação
    output_img = cv2.warpPerspective(image1, H_translation.dot(transformation_matrix),
                                     (x_max - x_min, y_max - y_min))

    # Sobrepor a imagem2 na imagem transformada
    output_img[translation_dist[1]:h2 + translation_dist[1],
               translation_dist[0]:w2 + translation_dist[0]] = image2

    # Exibir a imagem resultante
    cv2.imshow("Imagem Combinada", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    raise AssertionError("Não há keypoints suficientes.")
