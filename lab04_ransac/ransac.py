""" 
Inspirado em: https://en.wikipedia.org/wiki/Random_sample_consensus
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
from copy import copy
from numpy.random import default_rng

rng = default_rng()

class RANSAC:
    def __init__(self, n=10, k=100, t=0.05, d=10, model=None, loss=None, metric=None):
        self.n = n              # `n`: Minimum number of data points to estimate parameters
        self.k = k              # `k`: Maximum iterations allowed
        self.t = t              # `t`: Threshold value to determine if points are fit well
        self.d = d              # `d`: Number of close data points required to assert model fits well
        self.model = model      # `model`: class implementing `fit` and `predict`
        self.loss = loss        # `loss`: function of `y_true` and `y_pred` that returns a vector
        self.metric = metric    # `metric`: function of `y_true` and `y_pred` and returns a float
        self.best_fit = None
        self.best_error = np.inf

    # Fit the model to the data
    def fit(self, X, y):
        for _ in range(self.k):
            ids = rng.permutation(X.shape[0]) # Randomly shuffle the data points

            maybe_inliers = ids[: self.n] # Select the first `n` data points
            maybe_model = copy(self.model).fit(X[maybe_inliers], y[maybe_inliers]) 

            thresholded = ( 
                self.loss(y[ids][self.n :], maybe_model.predict(X[ids][self.n :])) 
                < self.t 
            )

            inlier_ids = ids[self.n :][np.flatnonzero(thresholded).flatten()] 

            if inlier_ids.size > self.d: 
                inlier_points = np.hstack([maybe_inliers, inlier_ids]) 
                better_model = copy(self.model).fit(X[inlier_points], y[inlier_points]) 

                this_error = self.metric(
                    y[inlier_points], better_model.predict(X[inlier_points])
                )

                if this_error < self.best_error:
                    self.best_error = this_error
                    self.best_fit = better_model

        return self

    def predict(self, X):
        return self.best_fit.predict(X) 

def square_error_loss(y_true, y_pred):
    return (y_true - y_pred) ** 2

def mean_square_error(y_true, y_pred):
    return np.sum(square_error_loss(y_true, y_pred)) / y_true.shape[0]

# Linear regression model
class LinearRegressor: 
    def __init__(self):
        self.params = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        r, _ = X.shape
        X = np.hstack([np.ones((r, 1)), X])
        self.params = np.linalg.inv(X.T @ X) @ X.T @ y
        return self

    # Predict the value of y for a given X
    def predict(self, X: np.ndarray):
        r, _ = X.shape
        X = np.hstack([np.ones((r, 1)), X])
        return X @ self.params

def detect_corners(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print('Não foi possível abrir ou encontrar a imagem')
        return None, None

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray_image)
    harris_corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    img[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]

    corners = np.argwhere(harris_corners > 0.01 * harris_corners.max())
    return corners, img

if __name__ == "__main__":
    image_path = sys.argv[1]
    corners, detected_image = detect_corners(image_path)

    if corners is not None:
        corners = corners[:, [1, 0]]  # Swap columns to get (x, y)
        X = corners[:, 0].reshape(-1, 1)
        y = corners[:, 1].reshape(-1, 1)

        regressor = RANSAC(model=LinearRegressor(), loss=square_error_loss, metric=mean_square_error)
        regressor.fit(X, y)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
        
        #Overlay the detected corners
        ax.scatter(corners[:, 0], corners[:, 1], c="red", s=1)

        if regressor.best_fit:
            # Ajustar os valores de line_x para ficarem dentro do intervalo de X (limites da imagem)
            line_x = np.linspace(min(X), max(X), num=100).reshape(-1, 1)
            
            # Garantir que o line_x esteja dentro dos limites da imagem
            line_x = np.clip(line_x, 0, detected_image.shape[1])  # Restringe aos limites da largura da imagem
            line_y = regressor.predict(line_x)
            
            # Garantir que o line_y também esteja dentro dos limites da imagem
            line_y = np.clip(line_y, 0, detected_image.shape[0])  # Restringe aos limites da altura da imagem
            
            # Plotar a linha dentro dos limites da imagem
            ax.plot(line_x, line_y, c="peru", linewidth=2)


        ax.set_title('Cantos Detectados e Modelo RANSAC')
        ax.axis('off')
        plt.show()
