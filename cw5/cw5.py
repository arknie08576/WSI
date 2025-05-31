import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("cw5/wsi5-25L_dataset.csv")
X = data.drop("quality", axis=1)
y = data["quality"]

# Standaryzacja
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Podział na train/val/test
X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
import numpy as np

class MLP:
    def __init__(self, layers, activation='relu', loss='mse', learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate
        self.activation = activation
        self.loss = loss
        self.weights = []
        self.biases = []

        for i in range(len(layers)-1):
            self.weights.append(np.random.randn(layers[i], layers[i+1]) * np.sqrt(2. / layers[i]))
            self.biases.append(np.zeros((1, layers[i+1])))

    def _activation(self, x, derivative=False):
        if self.activation == 'relu':
            return np.where(x > 0, x, 0) if not derivative else np.where(x > 0, 1, 0)
        elif self.activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig if not derivative else sig * (1 - sig)
        elif self.activation == 'tanh':
            return np.tanh(x) if not derivative else 1 - np.tanh(x) ** 2

    def _loss(self, y_true, y_pred, derivative=False):
        if self.loss == 'mse':
            return np.mean((y_true - y_pred) ** 2) if not derivative else (y_pred - y_true)

    def forward(self, X):
        self.z = []
        self.a = [X]
        for i in range(len(self.weights)):
            z = self.a[-1].dot(self.weights[i]) + self.biases[i]
            self.z.append(z)
            self.a.append(self._activation(z))
        return self.a[-1]

    def backward(self, y_true):
        deltas = [self._loss(y_true, self.a[-1], derivative=True) * self._activation(self.z[-1], derivative=True)]
        for i in reversed(range(len(self.weights)-1)):
            deltas.append(deltas[-1].dot(self.weights[i+1].T) * self._activation(self.z[i], derivative=True))
        deltas.reverse()

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.a[i].T.dot(deltas[i])
            self.biases[i] -= self.learning_rate * np.mean(deltas[i], axis=0, keepdims=True)

    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            self.backward(y)
            if epoch % 10 == 0:
                loss_val = self._loss(y, y_pred)
                print(f"Epoch {epoch}, Loss: {loss_val:.4f}")

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix

encoder = OneHotEncoder(sparse_output=False)
y_train_encoded = encoder.fit_transform(y_train.values.reshape(-1, 1))

model = MLP(layers=[X_train.shape[1], 64, 32, y_train_encoded.shape[1]], activation='relu', learning_rate=0.01)
model.fit(X_train, y_train_encoded, epochs=100)

# Predykcja
y_pred_probs = model.forward(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_test.values

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

