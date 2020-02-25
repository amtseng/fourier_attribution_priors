import torch
import numpy as np

print("Creating neural network")
device = torch.device("cuda")

X = torch.randn(100, 16).cuda()
y = torch.randn(100, 1).cuda()

weights_1 = torch.randn(16, 5).cuda()
biases_1 = torch.randn(1, 5).cuda()
weights_2 = torch.randn(5, 1).cuda()
biases_2 = torch.randn(1, 1).cuda()

weights_1.requires_grad = True
biases_1.requires_grad = True
weights_2.requires_grad = True
biases_2.requires_grad = True

sigmoid = torch.nn.Sigmoid()

print("Performing forward prop")
h = sigmoid(torch.torch.matmul(X, weights_1) + biases_1)
y_hat = sigmoid(torch.torch.matmul(h, weights_2) + biases_2)
loss = torch.mean((y - y_hat) * (y - y_hat))
print("\tLoss: %f" % loss.item())

print("Performing back prop")
learning_rate = 1
loss.backward()
with torch.no_grad():
    weights_1 -= learning_rate * weights_1.grad
    weights_2 -= learning_rate * weights_2.grad
    biases_1 -= learning_rate * biases_1.grad
    biases_2 -= learning_rate * biases_2.grad

h = sigmoid(torch.torch.matmul(X, weights_1) + biases_1)
y_hat = sigmoid(torch.torch.matmul(h, weights_2) + biases_2)
loss = torch.mean((y - y_hat) * (y - y_hat))
print("\tLoss: %f" % loss.item())
