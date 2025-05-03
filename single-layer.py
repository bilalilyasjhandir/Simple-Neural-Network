import numpy as np

#sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#sigmoid derivative for backprop
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

#binary cross-entropy loss
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-8  #to avoid log(0)
    return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

#generate toy dataset (here is OR logic gate)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])  #4 samples 2 features

y = np.array([[0], [1], [1], [1]])  #OR gate output

#initialize weights and bias
np.random.seed(42)
W = np.random.randn(2, 1)
b = np.random.randn(1)

#training parameters
lr = 0.1
epochs = 1000

#training loop
for epoch in range(epochs):
    #forward pass
    z = np.dot(X, W) + b
    y_pred = sigmoid(z)

    #loss
    loss = binary_cross_entropy(y, y_pred)

    #backward pass
    dz = y_pred - y  #derivative of loss wrt z
    dW = np.dot(X.T, dz) / len(X)
    db = np.mean(dz)

    #update weights
    W -= lr * dW
    b -= lr * db

    #print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

#final predictions
print("\nFinal Predictions:")
print(sigmoid(np.dot(X, W) + b).round())