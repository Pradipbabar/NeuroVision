# Module 5: Deep Learning

Welcome to Module 5 of the NeuroVision roadmap! This module introduces you to the fascinating world of deep learning, a subset of machine learning that utilizes neural networks to model and solve complex problems.

---

## **Topics Covered**

1. **Introduction to Neural Networks**
    - What are Neural Networks?
    - Components of a Neural Network (Neurons, Layers, Weights, and Biases)
    - Activation Functions (ReLU, Sigmoid, Tanh, Softmax)
2. **Deep Learning Frameworks**
    - Overview of TensorFlow and PyTorch
    - Installation and Basic Setup
3. **Architectures in Deep Learning**
    - Convolutional Neural Networks (CNNs)
    - Recurrent Neural Networks (RNNs)
    - Autoencoders
4. **Transfer Learning**
    - Concept and Benefits
    - Pre-trained Models (ResNet, VGG, BERT)
5. **Training Deep Neural Networks**
    - Loss Functions and Optimizers
    - Regularization Techniques
    - Batch Normalization and Dropout
6. **Practical Implementation**
    - Hands-on with TensorFlow and PyTorch
    - Building and Training Neural Networks

---

## **1. Introduction to Neural Networks**

### What are Neural Networks?
- Neural networks are computational models inspired by the human brain, consisting of interconnected layers of artificial neurons.

### Components of a Neural Network:
- **Neurons:** Basic units that process inputs to produce outputs.
- **Layers:** Input layer, hidden layers, and output layer.
- **Weights and Biases:** Parameters adjusted during training.

### Activation Functions:
- Add non-linearity to the model.
    - **ReLU:** `max(0, x)`
    - **Sigmoid:** `1 / (1 + exp(-x))`
    - **Tanh:** `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`
    - **Softmax:** Converts outputs into probabilities.

---

## **2. Deep Learning Frameworks**

### TensorFlow:
- A comprehensive library for deep learning.
```python
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

### PyTorch:
- A flexible and dynamic framework for deep learning.
```python
import torch
import torch.nn as nn
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
model = NeuralNet()
```

---

## **3. Architectures in Deep Learning**

### Convolutional Neural Networks (CNNs):
- Specialized for image data.
- Key components: Convolutional layers, Pooling layers, and Fully connected layers.

### Recurrent Neural Networks (RNNs):
- Designed for sequential data like time series or text.
- Variants include LSTMs and GRUs.

### Autoencoders:
- Used for unsupervised learning tasks like dimensionality reduction and anomaly detection.

---

## **4. Transfer Learning**

### Concept:
- Leveraging pre-trained models to solve new but similar tasks.

### Popular Pre-trained Models:
- ResNet, VGG for image tasks.
- BERT for NLP tasks.
```python
from tensorflow.keras.applications import ResNet50
model = ResNet50(weights='imagenet')
```

---

## **5. Training Deep Neural Networks**

### Loss Functions:
- Measure how well the model's predictions match the target.
    - Examples: Mean Squared Error, Cross-Entropy Loss.

### Optimizers:
- Algorithms that adjust weights during training.
    - Examples: SGD, Adam, RMSprop.

### Regularization Techniques:
- Prevent overfitting by penalizing complex models.
    - L1/L2 Regularization, Dropout.

### Batch Normalization:
- Stabilizes and accelerates training.

---

## **6. Practical Implementation**

### Building a Simple Neural Network in TensorFlow:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Building a CNN in PyTorch:
```python
import torch.nn as nn
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, activation=F.relu)
        self.fc1 = nn.Linear(32*26*26, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 32*26*26)
        x = self.fc1(x)
        return x
```

