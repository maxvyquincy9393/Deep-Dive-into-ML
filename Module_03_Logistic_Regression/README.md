# Module 03: Logistic Regression

**Part of the Deep Dive Machine Learning Series** | Based on **Stanford CS229**

From regression to classification - learning decision boundaries with the sigmoid function and cross-entropy loss.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![CS229](https://img.shields.io/badge/Stanford-CS229-red.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.20+-green.svg)

## Overview

Logistic Regression is the workhorse of binary classification. Despite its name, it's a classification algorithm that models the probability of class membership using the sigmoid function. This module covers everything from the math to practical implementation.

## What You'll Learn

- **Sigmoid Function**: Squashing outputs to probabilities $\sigma(z) = \frac{1}{1+e^{-z}}$
- **Cross-Entropy Loss**: Why MSE doesn't work for classification
- **Maximum Likelihood**: Probabilistic interpretation of logistic regression
- **Decision Boundary**: Where $P(y=1|x) = 0.5$
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Regularization**: L2 penalty for logistic regression

## Notebooks

| Notebook | Description |
|----------|-------------|
| `Module_03_Logistic_Regression_Scratch.ipynb` | From-scratch implementation |
| `Module_03b_Advanced_Logistic.ipynb` | Advanced topics and optimization |

## Additional Resources

- `Evaluation_Metrics_Quick_Reference.md` - Quick reference for all metrics
- `Logistic_Regression_Cheat_Sheet.md` - Formula cheat sheet

## Real-World Case Study

**Heart Disease Classification** - Predict heart disease presence based on clinical measurements.

## Key Equations

**Sigmoid Function:**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Hypothesis:**
$$h_\theta(x) = \sigma(\theta^T x) = P(y=1|x;\theta)$$

**Cross-Entropy Loss:**
$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1-y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]$$

**Gradient:**
$$\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$$

## CS229 Connection

- **Lecture 3**: Logistic Regression
- **Lecture 4**: Generalized Linear Models (Logistic as GLM)

## Related Modules

| Previous | Current | Next |
|----------|---------|------|
| [Module 02: Advanced Linear Regression](../Module_02_Advanced_Linear_Regression/) | **Module 03: Logistic Regression** | [Module 04: Softmax Regression](../Module_04_Softmax_Regression/) |

## Author

**Kanisius Bagas**

- GitHub: [@maxvyquincy9393](https://github.com/maxvyquincy9393)
- LinkedIn: [kanisiusbagas1212](https://www.linkedin.com/in/kanisiusbagas1212)
- Email: maxvy1218@gmail.com

---

**[← Back to Main Course](../README.md)**
