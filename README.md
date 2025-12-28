# TinyTorch

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**TinyTorch** is a minimal deep learning framework built from scratch in Python.  
It implements automatic differentiation, tensor operations, linear layers, multi-layer perceptrons (MLPs), activation functions, and training loops.  

This project is designed for educational purposes to help understand the inner workings of neural networks and autograd systems.

---

## Features

- **Tensor class** with gradient tracking
- **Autograd engine** for automatic differentiation
- **Linear layers and MLPs**
- **Common activation functions**: ReLU, Leaky ReLU (LU), Softmax
- **Loss functions**: CrossEntropyWithSoftmax
- **Optimizers**: SGD, with zero_grad support
- **Simple training loop** with batch processing
- **Visualization**: Decision boundary plots (optional GUI)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Arian-jafari/TinyTorch.git
cd TinyTorch
```
Install dependencies:

```bash
pip install numpy matplotlib
```
No deep learning frameworks are required — everything is built from scratch.

## Quick Example

```bash
from module import MLP
from tensor import Tensor
from optimizer import SGD
from loss import CrossEntropyWithSoftmax
from activations import LU

# Create dataset (example)
X = Tensor([[0,0],[0,1],[1,0],[1,1]])
y = Tensor([0,1,1,0])  # example labels

# Define model
model = MLP(num_features=2, num_classes=2)
optimizer = SGD(model.parameters(), lr=0.01)
loss_fn = CrossEntropyWithSoftmax

# Training loop
for epoch in range(100):
    logits = model(X)
    loss = loss_fn.apply(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")
```

## Project Structure

```
TinyTorch/
│
├── tensor.py # Tensor implementation with autograd
├── function.py # Base Function class and operations
├── module.py # Base Module and Linear layer
├── activations.py # ReLU, LU, Softmax implementations
├── loss.py # CrossEntropy loss
├── optimizer.py # SGD and gradient utilities
├── example.py # Sample training scripts
├── gui.py # Optional visualization GUI
└── README.md
```

### Notes

- Everything is implemented from scratch, without PyTorch or TensorFlow.
- Focuses on clarity, understanding, and hands-on experimentation.
- Useful for learning the fundamentals of neural networks, backpropagation, and autograd.

