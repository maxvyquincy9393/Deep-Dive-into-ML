# Module 01: Linear Regression

**Part of the Deep Dive Machine Learning Series** | Based on **Stanford CS229**

The foundation of machine learning - understanding how to fit a line (or hyperplane) to data using gradient descent and the normal equation.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![CS229](https://img.shields.io/badge/Stanford-CS229-red.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.20+-green.svg)

## Overview

Linear Regression is where every ML journey begins. This module covers the complete mathematical foundation and implementation of linear regression, from the simple y = mx + b to multivariate regression with gradient descent optimization.

## What You'll Learn

- **Hypothesis Function**: How $h_\theta(x) = \theta^T x$ represents our model
- **Cost Function**: Mean Squared Error (MSE) and why we use it
- **Gradient Descent**: Iterative optimization to minimize cost
- **Normal Equation**: Closed-form solution $\theta = (X^T X)^{-1} X^T y$
- **Feature Scaling**: Why and how to normalize features
- **Learning Rate**: How to choose α and diagnose convergence

## Notebooks

| Notebook | Description |
|----------|-------------|
| `Module_01_Linear_Regression.ipynb` | Core concepts, derivations, and implementation |
| `Module_01b_SGD_NormalEquation.ipynb` | Stochastic Gradient Descent vs Normal Equation comparison |

## Real-World Case Study

**California Housing Prices** - Predict median house values based on location, demographics, and housing characteristics.

## Key Equations

**Hypothesis:**
$$h_\theta(x) = \theta_0 + \theta_1 x_1 + ... + \theta_n x_n = \theta^T x$$

**Cost Function (MSE):**
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

**Gradient Descent Update:**
$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$$

**Normal Equation:**
$$\theta = (X^T X)^{-1} X^T y$$

## CS229 Connection

- **Lecture 2**: Linear Regression, Gradient Descent
- **Lecture 3**: Normal Equation, Probabilistic Interpretation

## Related Modules

| Previous | Current | Next |
|----------|---------|------|
| - | **Module 01: Linear Regression** | [Module 02: Advanced Linear Regression](../Module_02_Advanced_Linear_Regression/) |

## Author

**Kanisius Bagas**

- GitHub: [@maxvyquincy9393](https://github.com/maxvyquincy9393)
- LinkedIn: [kanisiusbagas1212](https://www.linkedin.com/in/kanisiusbagas1212)
- Email: maxvy1218@gmail.com

---

**[← Back to Main Portfolio](../README.md)**
