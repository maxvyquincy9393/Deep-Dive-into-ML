# Module 06: Newton's Method & Perceptron

**Part of the Deep Dive Machine Learning Series** | Based on **Stanford CS229**

Second-order optimization and the foundational classification algorithm.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![CS229](https://img.shields.io/badge/Stanford-CS229-red.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.20+-green.svg)

## Overview

This module covers two important topics: Newton's Method for faster optimization and the Perceptron algorithm as a foundational classifier. Newton's Method uses second-order derivatives (Hessian) to achieve faster convergence than gradient descent.

## What You'll Learn

### Newton's Method
- **Hessian Matrix**: Second-order partial derivatives
- **Newton-Raphson Update**: Using curvature information
- **Faster Convergence**: Quadratic vs linear convergence
- **Fisher Scoring**: Newton's method for logistic regression

### Perceptron
- **Perceptron Algorithm**: The simplest neural network
- **Linear Separability**: When perceptron converges
- **Perceptron Convergence Theorem**: Guaranteed convergence for separable data
- **Limitations**: Why perceptron fails on XOR

## Notebooks

| Notebook | Description |
|----------|-------------|
| `Module_06_Logistic_Regression_Newton_Method.ipynb` | Newton's method for logistic regression |
| `Module_06_1_Perceptron.ipynb` | Perceptron algorithm implementation |

## Real-World Case Study

**Binary Classification Comparison** - Comparing gradient descent vs Newton's method vs Perceptron on classification tasks.

## Key Equations

**Newton's Method Update:**
$$\theta := \theta - H^{-1} \nabla_\theta J(\theta)$$

**Hessian Matrix:**
$$H_{jk} = \frac{\partial^2 J}{\partial \theta_j \partial \theta_k}$$

**Perceptron Update Rule:**
$$\theta := \theta + \alpha (y^{(i)} - \hat{y}^{(i)}) x^{(i)}$$

**Convergence:**
- Gradient Descent: Linear convergence
- Newton's Method: Quadratic convergence (when near optimum)

## CS229 Connection

- **Lecture 3**: Newton's Method
- **Lecture Notes**: Optimization methods

## Comparison

| Method | Convergence | Cost per Iteration | Memory |
|--------|-------------|-------------------|--------|
| Gradient Descent | O(1/k) | O(n) | O(n) |
| Newton's Method | O(1/k²) | O(n³) | O(n²) |
| Perceptron | Finite (if separable) | O(n) | O(n) |

## Related Modules

| Previous | Current | Next |
|----------|---------|------|
| [Module 05: LWLR](../Module_05_LWLR/) | **Module 06: Newton's Method & Perceptron** | [Module 07: GLM](../Module_07_GLM/) |

## Author

**Kanisius Bagas**

- GitHub: [@maxvyquincy9393](https://github.com/maxvyquincy9393)
- LinkedIn: [kanisiusbagas1212](https://www.linkedin.com/in/kanisiusbagas1212)
- Email: maxvy1218@gmail.com

---

**[← Back to Main Portfolio](../README.md)**
