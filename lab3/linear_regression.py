import numpy as np

class LinearRegression:
    def __init__(self):
        self._W = None
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        _, D = X.shape
        X = np.column_stack((np.ones(X.shape[0]), X))
        self._W = np.zeros(D + 1, )
        cov = X.T @ X 
        inv_cov = np.linalg.inv(cov)  
        self._W = inv_cov @ X.T @ y  

    def predict(self, X:np.ndarray) -> np.ndarray:
        X = np.column_stack((np.ones(X.shape[0]), X))
        y_hat = X @ self._W 
        return y_hat

    def evaluate(self, y: np.ndarray, y_hat: np.ndarray):
        rmse = ((y - y_hat) ** 2).mean()**0.5
        return rmse