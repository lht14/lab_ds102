import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score


class LogisticRegression:
    def __init__(self, epoch: int, lr: float):
        self.epoch = epoch
        self.lr = lr
        self.w = None
        self.losses = []

    def loss_fn(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        l = (1 - y) * np.log(1 - y_hat + 10e-15) + y * np.log(y_hat + 10e-15)
        return -l.mean()

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, d = X.shape
        self.w = np.zeros((d,), dtype=np.float64)
        self.b = 0.0
        e = 0
        with tqdm(range(self.epoch), desc=f"Epoch {e+1}:") as pbar:
            for _ in pbar:
                # forward pass
                y_hat = self.predict(X)
                deltal_y = y_hat - y  # (N, 1)

                # backward pass
                gradient = deltal_y.T @ X   #(1, d)
                self.w = self.w - (1/N)*self.lr*gradient.T  # (d, 1)
                self.b = self.b - self.lr * deltal_y.mean()

                l = self.loss_fn(y, y_hat)
                self.losses.append(l)
                
                pbar.set_postfix({"loss": l})
            e += 1

    def sigmoid(self, z: np.ndarray):
        return 1 / (1 + np.exp(-z))

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.w + self.b
        y_hat = self.sigmoid(z)
        return y_hat

    def evaluate(self, y, y_hat) -> dict:
        y_hat = np.where(y_hat <= 0.5, 0, 1)
        precision = precision_score(y, y_hat)
        recall = recall_score(y, y_hat)
        f1 = f1_score(y, y_hat)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }