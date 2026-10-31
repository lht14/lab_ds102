import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

class SoftmaxRegression:
    def __init__(self, epoch: int, lr: float):
        self.epoch = epoch
        self.lr = lr
        self.losses = []

    def loss_fn(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        return -(y * np.log(y_hat + 10e-15)).sum(axis=-1).mean()

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, d = X.shape
        _, k = y.shape
        self.w = np.zeros((d, k))
        self.b = np.zeros(k)
        e = 0
        with tqdm(range(self.epoch)) as pbar:
            for _ in pbar:
                y_hat = self.predict(X)

                delta_y = (y_hat - y)
                gradient = X.T @ delta_y
                self.w = self.w - (1/N) * self.lr * gradient
                self.b = self.b - self.lr * delta_y.mean(axis=0)

                l = self.loss_fn(y, y_hat)
                self.losses.append(l)
                
                pbar.set_postfix({
                    "Epoch": e+1,
                    "Loss": l
                })
                e += 1

    def softmax(self, z: np.ndarray):
        denum = np.exp(z).sum(axis=1, keepdims=True)
        return np.exp(z) / denum

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.w + self.b
        y_hat = self.softmax(z)
        return y_hat

    def evaluate(self, y, y_hat) -> dict:
        y_hat = np.argmax(y_hat, axis=-1)
        precision = precision_score(y, y_hat, average="macro")
        recall = recall_score(y, y_hat, average="macro")
        f1 = f1_score(y, y_hat, average="macro")
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }