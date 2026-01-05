# Module 05: Locally Weighted Linear Regression (LWLR)

**Part of the Deep Dive Machine Learning Series** | Based on **Stanford CS229**

A non-parametric approach that fits local models weighted by proximity to the query point.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![CS229](https://img.shields.io/badge/Stanford-CS229-red.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.20+-green.svg)

## Overview

Sometimes a global linear model isn't enough. Locally Weighted Linear Regression (LWLR) fits a different linear model for each prediction, giving more weight to nearby training points. It's a powerful non-parametric method that can capture complex non-linear relationships.

## What You'll Learn

- **Parametric vs Non-Parametric**: Fixed parameters vs keeping training data
- **Gaussian Kernel Weighting**: How proximity affects weights
- **Bandwidth Selection**: The τ parameter and its effect on bias-variance
- **Computational Cost**: Why LWLR is expensive at prediction time
- **When to Use LWLR**: Scenarios where non-parametric methods shine
- **Loess/Lowess**: Robust locally weighted regression

## Notebooks

| Notebook | Description |
|----------|-------------|
| `LWLR_Tutorial_Complete.ipynb` | Step-by-step tutorial |
| `Module_05_Advanced_LWLR.ipynb` | Advanced topics and bandwidth selection |

## Real-World Case Study

**Temperature Prediction** - Predict temperature based on various atmospheric measurements with locally adaptive models.

## Key Equations

**Weight Function (Gaussian Kernel):**
$$w^{(i)} = \exp\left(-\frac{(x^{(i)} - x)^2}{2\tau^2}\right)$$

**Weighted Cost Function:**
$$J(\theta) = \sum_{i=1}^{m} w^{(i)} (y^{(i)} - \theta^T x^{(i)})^2$$

**Closed-Form Solution:**
$$\theta = (X^T W X)^{-1} X^T W y$$

where W is diagonal matrix with $W_{ii} = w^{(i)}$

## CS229 Connection

- **Lecture 3**: Locally Weighted Regression
- **Lecture Notes**: Non-parametric Methods

## Key Insights

| Bandwidth (τ) | Effect |
|---------------|--------|
| Small τ | High variance, low bias (overfitting risk) |
| Large τ | Low variance, high bias (approaches global linear) |

## Related Modules

| Previous | Current | Next |
|----------|---------|------|
| [Module 04: Softmax Regression](../Module_04_Softmax_Regression/) | **Module 05: LWLR** | [Module 06: Newton's Method](../Module_06_Newton_Perceptron/) |

## Author

**Kanisius Bagas**

- GitHub: [@maxvyquincy9393](https://github.com/maxvyquincy9393)
- LinkedIn: [kanisiusbagas1212](https://www.linkedin.com/in/kanisiusbagas1212)
- Email: maxvy1218@gmail.com

---

**[← Back to Main Portfolio](../README.md)**
