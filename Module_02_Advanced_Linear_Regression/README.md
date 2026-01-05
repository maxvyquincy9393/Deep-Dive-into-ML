# Module 02: Advanced Linear Regression

**Part of the Deep Dive Machine Learning Series** | Based on **Stanford CS229**

Taking linear regression further with regularization, polynomial features, and cross-validation techniques.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![CS229](https://img.shields.io/badge/Stanford-CS229-red.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.20+-green.svg)

## Overview

Real-world data rarely fits a simple line. This module explores how to handle complex relationships through feature engineering, prevent overfitting with regularization, and validate models properly with cross-validation.

## What You'll Learn

- **Polynomial Features**: Capturing non-linear relationships with linear models
- **Ridge Regression (L2)**: Adding $\lambda \|\theta\|_2^2$ penalty to prevent overfitting
- **Lasso Regression (L1)**: Sparse solutions with $\lambda \|\theta\|_1$ penalty
- **Elastic Net**: Combining L1 and L2 regularization
- **Bias-Variance Tradeoff**: Understanding underfitting vs overfitting
- **Cross-Validation**: K-fold CV for model selection

## Notebooks

| Notebook | Description |
|----------|-------------|
| `Module_02_Advanced_Linear_Regression_Scratch.ipynb` | Complete implementation with regularization |

## Real-World Case Study

**Medical Cost Prediction** - Predict individual medical costs based on age, BMI, smoking status, and other factors.

## Key Equations

**Ridge Regression (L2):**
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^{n} \theta_j^2$$

**Lasso Regression (L1):**
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^{n} |\theta_j|$$

**Closed-form Ridge Solution:**
$$\theta = (X^T X + \lambda I)^{-1} X^T y$$

## CS229 Connection

- **Lecture 4**: Regularization
- **Lecture Notes**: Regularization and Model Selection

## Related Modules

| Previous | Current | Next |
|----------|---------|------|
| [Module 01: Linear Regression](../Module_01_Linear_Regression/) | **Module 02: Advanced Linear Regression** | [Module 03: Logistic Regression](../Module_03_Logistic_Regression/) |

## Author

**Kanisius Bagas**

- GitHub: [@maxvyquincy9393](https://github.com/maxvyquincy9393)
- LinkedIn: [kanisiusbagas1212](https://www.linkedin.com/in/kanisiusbagas1212)
- Email: maxvy1218@gmail.com

---

**[← Back to Main Portfolio](../README.md)**
