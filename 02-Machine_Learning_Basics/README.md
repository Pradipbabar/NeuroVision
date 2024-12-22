# Module 2: Machine Learning Basics

Welcome to Module 2 of the NeuroVision roadmap! In this module, we’ll dive into the foundational concepts of Machine Learning (ML), a crucial subset of Artificial Intelligence. By the end of this module, you will have a solid understanding of the types of Machine Learning, essential algorithms, and key evaluation metrics.

---

## **Topics Covered**

1. **Introduction to Machine Learning**
2. **Types of Machine Learning**
    - Supervised Learning
    - Unsupervised Learning
    - Reinforcement Learning
3. **Basic Algorithms**
    - Linear Regression
    - K-Nearest Neighbors (KNN)
4. **Evaluation Metrics**
    - Accuracy
    - Precision
    - Recall
    - F1 Score

---

## **1. Introduction to Machine Learning**

Machine Learning enables systems to learn and improve from experience without explicit programming. By leveraging data, algorithms allow computers to identify patterns, make predictions, and optimize decisions autonomously.

Key Characteristics:
- **Data-Driven:** Relies on data for learning.
- **Generalization:** Applies learned patterns to unseen data.
- **Automation:** Minimizes human intervention in repetitive tasks.

Applications:
- Email spam detection
- Voice assistants
- Predictive analytics

---

## **2. Types of Machine Learning**

### **Supervised Learning**

In supervised learning, the algorithm is trained on labeled data where the input-output relationship is explicitly defined. The goal is to map inputs to outputs accurately.

Examples:
- Classification (e.g., spam detection)
- Regression (e.g., predicting house prices)

Popular Algorithms:
- Linear Regression
- Logistic Regression
- Support Vector Machines (SVMs)

---

### **Unsupervised Learning**

Unsupervised learning works with unlabeled data, aiming to uncover hidden patterns or structures within the dataset.

Examples:
- Clustering (e.g., customer segmentation)
- Dimensionality Reduction (e.g., Principal Component Analysis)

Popular Algorithms:
- K-Means Clustering
- Hierarchical Clustering
- DBSCAN

---

### **Reinforcement Learning**

In reinforcement learning, an agent learns by interacting with an environment, receiving rewards or penalties for actions.

Examples:
- Game playing (e.g., AlphaGo)
- Robotics

Key Concepts:
- Agent, Environment, Reward, Policy
- Exploration vs. Exploitation

Popular Algorithms:
- Q-Learning
- Deep Q-Networks (DQN)

---

## **3. Basic Algorithms**

### **Linear Regression**
A statistical method for modeling relationships between a dependent variable and one or more independent variables. It predicts continuous values.

Formula:
\[
y = \beta_0 + \beta_1 x + \epsilon
\]
Where:
- \( \beta_0 \): Intercept
- \( \beta_1 \): Slope
- \( \epsilon \): Error term

Applications:
- Predicting house prices
- Sales forecasting

---

### **K-Nearest Neighbors (KNN)**
A simple algorithm that stores all available cases and predicts the output based on a similarity measure (e.g., distance metrics like Euclidean).

Steps:
1. Choose the number of neighbors (\(k\)).
2. Calculate the distance between data points.
3. Assign the label based on the majority of neighbors.

Applications:
- Handwritten digit recognition
- Recommender systems

---

## **4. Evaluation Metrics**

### **Accuracy**
The ratio of correctly predicted observations to the total observations.
\[
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
\]
Where:
- TP: True Positives
- TN: True Negatives
- FP: False Positives
- FN: False Negatives

---

### **Precision**
The ratio of correctly predicted positive observations to the total predicted positives.
\[
Precision = \frac{TP}{TP + FP}
\]

---

### **Recall**
The ratio of correctly predicted positive observations to the all observations in actual class.
\[
Recall = \frac{TP}{TP + FN}
\]

---

### **F1 Score**
The weighted average of Precision and Recall, providing a balance between the two.
\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

---