import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ----------------------
# 1. Create Simple Data
# ----------------------
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
y = np.array([3, 4, 7, 8, 11, 13, 15, 18, 20])

# ----------------------
# 2. Custom Linear Regression
# ----------------------
class CustomLinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weight = 0
        self.bias = 0

    def fit(self, X, y):
        n = len(X)
        for _ in range(self.n_iters):
            y_pred = self.weight * X + self.bias
            dw = (-2/n) * np.sum(X * (y - y_pred))
            db = (-2/n) * np.sum(y - y_pred)
            self.weight -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return self.weight * X + self.bias

# Train custom model
custom_lr = CustomLinearRegression(lr=0.01, n_iters=10000)
custom_lr.fit(X.flatten(), y)
y_pred_custom = custom_lr.predict(X.flatten())

# ----------------------
# 3. Sklearn Linear Regression
# ----------------------
sklearn_lr = LinearRegression()
sklearn_lr.fit(X, y)
y_pred_sklearn = sklearn_lr.predict(X)

# ----------------------
# 4. Visualization
# ----------------------
plt.figure(figsize=(10,5))
plt.scatter(X, y, color="blue", label="Actual Data Points")

plt.plot(X, y_pred_custom, color="red",
         label=f"Custom Model Line (w={custom_lr.weight:.2f}, b={custom_lr.bias:.2f})")

plt.plot(X, y_pred_sklearn, color="green", linestyle="--",
         label=f"Scikit-learn Model Line (w={sklearn_lr.coef_[0]:.2f}, b={sklearn_lr.intercept_:.2f})")

plt.xlabel("Feature (X)")
plt.ylabel("Target (y)")
plt.title("Linear Regression: Custom vs Scikit-learn")
plt.legend()
plt.show()
