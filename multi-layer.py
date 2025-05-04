import numpy as np

#sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#sigmoid derivative
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

#hyperparameters
input_size = 2
hidden_size = 4
output_size = 1
lr = 0.1
epochs = 10000

#initialize weights and bias
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

#training loop
for epoch in range(epochs):
    #forward pass
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    y_pred = sigmoid(Z2)
    
    #loss
    loss = binary_cross_entropy(y, y_pred)
    
    #backward pass
    dZ2 = y_pred - y  #(4, 1)
    dW2 = np.dot(A1.T, dZ2) / X.shape[0]
    db2 = np.mean(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / X.shape[0]
    db1 = np.mean(dZ1, axis=0, keepdims=True)
    
    #update weights
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    #print loss every 1000 steps
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

#final predictions
print("\nFinal predictions:")
print(sigmoid(np.dot(sigmoid(np.dot(X, W1) + b1), W2) + b2).round())