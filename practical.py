import numpy as np
def sigmoid(x):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (str): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(-x)
    return 1 / (1 + e)


X = np.array([[1, 1, 1], [1, 2, 1], [0, 1, 0], [1, 3, 1]])
y = np.array([0, 1, 0, 1])
theta = np.array((1, 1, 1))
s = sigmoid(np.matmul(theta, X.T))

print(np.matmul(theta, X.T))
print(np.dot(theta, X.T))
print(s)
a = -np.multiply(y, np.log(s))
print(a)
b = - np.multiply((1 - y), np.log(1 - s))
print(b)
print(np.sum(a + b)/len(X))
print(np.mean(a+b))

