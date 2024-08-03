# Basics of Calculus

Calculus is a branch of mathematics that studies how things change. It is divided into two main areas: differential calculus and integral calculus. Both are essential in various fields, including AI, machine learning, and data analysis.

## 1. Differential Calculus

Differential calculus focuses on the concept of a derivative, which measures how a function changes as its input changes.

## Derivatives

**Definition**: The derivative of a function measures the rate at which the function's output changes as its input changes. In simple terms, it gives the slope of the function at any given point.

**Example**: Consider the function \( f(x) = x^2 \).

To find the derivative \( f'(x) \):
- Apply the power rule: \( \frac{d}{dx} (x^n) = n \cdot x^{n-1} \).
- For \( f(x) = x^2 \):
  \[ f'(x) = 2 \cdot x^{2-1} = 2x \]

So, the derivative of \( x^2 \) is \( 2x \). This means that at any point \( x \), the rate of change of \( x^2 \) is \( 2x \). For example, at \( x = 3 \), the rate of change is \( 6 \).

## Applications

- **Optimization**: Derivatives are used to find local maxima and minima of functions, which is crucial for training machine learning models (e.g., finding the best parameters in gradient descent).
- **Rate of Change**: In AI, derivatives help understand how changes in input features affect the output predictions.

## 2. Integral Calculus

Integral calculus deals with the concept of an integral, which represents the accumulation of quantities and the area under curves.

## Integrals

**Definition**: The integral of a function represents the total accumulation of the quantity described by the function. It is the reverse process of differentiation.

**Example**: Consider the function \( f(x) = 2x \).

To find the integral of \( f(x) \):
- Apply the power rule for integration: \( \int x^n \, dx = \frac{x^{n+1}}{n+1} + C \), where \( C \) is the constant of integration.
- For \( f(x) = 2x \):
  \[ \int 2x \, dx = 2 \cdot \frac{x^{1+1}}{1+1} + C = x^2 + C \]

So, the integral of \( 2x \) is \( x^2 + C \). This represents the area under the curve \( f(x) = 2x \) from a given point to another.

## Applications

- **Area Under Curve**: Integrals can calculate areas under curves, which is useful for understanding distributions and probabilities in AI.
- **Cumulative Metrics**: In machine learning, integrals are used to accumulate errors over time or space.

## 3. Gradient Descent

**Definition**: Gradient descent is an optimization algorithm used to minimize the loss function in machine learning models. It involves calculating the gradient (derivative) of the loss function with respect to model parameters and updating the parameters in the direction that reduces the loss.

**Example**: Consider a simple linear regression problem where the goal is to minimize the mean squared error (MSE) between predicted and actual values.

- **Loss Function**: \( L(w) = \frac{1}{n} \sum_{i=1}^{n} (y_i - (w \cdot x_i + b))^2 \), where \( w \) and \( b \) are the parameters (weights and bias) to be optimized.

- **Gradient Calculation**: Compute the gradient of \( L(w) \) with respect to \( w \) and \( b \):
  \[ \frac{\partial L}{\partial w} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - (w \cdot x_i + b)) \cdot x_i \]
  \[ \frac{\partial L}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - (w \cdot x_i + b)) \]

- **Update Rule**: Adjust \( w \) and \( b \) using the gradients and a learning rate \( \eta \):
  \[ w := w - \eta \cdot \frac{\partial L}{\partial w} \]
  \[ b := b - \eta \cdot \frac{\partial L}{\partial b} \]

Gradient descent iteratively adjusts the parameters to minimize the loss function, eventually finding the best-fit line for the data.

## Summary

- **Differential Calculus**: Focuses on derivatives and how functions change. Used in optimization and understanding rates of change.
- **Integral Calculus**: Focuses on integrals and accumulation of quantities. Used for calculating areas and cumulative metrics.
- **Gradient Descent**: An optimization technique that uses derivatives to iteratively adjust model parameters to minimize the loss function.
