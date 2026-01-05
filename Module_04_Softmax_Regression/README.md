# Module 04: Softmax Regression

**Part of the Deep Dive Machine Learning Series** | Based on **Stanford CS229**

Extending binary classification to multiple classes with the softmax function.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![CS229](https://img.shields.io/badge/Stanford-CS229-red.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.20+-green.svg)

## Overview

When you have more than two classes, you need Softmax Regression (also called Multinomial Logistic Regression). This module extends the binary logistic regression framework to handle K classes, using the softmax function to produce a probability distribution over all classes.

## What You'll Learn

- **Softmax Function**: Converting K scores to probabilities that sum to 1
- **Cross-Entropy for Multiclass**: Generalizing binary cross-entropy
- **One-Hot Encoding**: Representing categorical labels
- **Multiclass Decision Boundaries**: How regions are partitioned
- **Numerical Stability**: Avoiding overflow in softmax computation
- **One-vs-Rest vs Softmax**: Comparing multiclass approaches

## Notebooks

| Notebook | Description |
|----------|-------------|
| `Module_04_Softmax_Regression.ipynb` | Complete implementation and theory |

## Real-World Case Study

**Wine Quality Classification** - Classify wines into quality categories based on chemical properties.

## Key Equations

**Softmax Function:**
$$P(y=k|x) = \frac{e^{\theta_k^T x}}{\sum_{j=1}^{K} e^{\theta_j^T x}}$$

**Cross-Entropy Loss (Multiclass):**
$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} \mathbf{1}\{y^{(i)}=k\} \log P(y^{(i)}=k|x^{(i)})$$

**Stable Softmax (subtract max):**
$$\text{softmax}(z)_k = \frac{e^{z_k - \max(z)}}{\sum_j e^{z_j - \max(z)}}$$

## CS229 Connection

- **Lecture 4**: Softmax Regression as a GLM
- **Lecture Notes**: Multiclass Classification

## Related Modules

| Previous | Current | Next |
|----------|---------|------|
| [Module 03: Logistic Regression](../Module_03_Logistic_Regression/) | **Module 04: Softmax Regression** | [Module 05: LWLR](../Module_05_LWLR/) |

## Author

**Kanisius Bagas**

- GitHub: [@maxvyquincy9393](https://github.com/maxvyquincy9393)
- LinkedIn: [kanisiusbagas1212](https://www.linkedin.com/in/kanisiusbagas1212)
- Email: maxvy1218@gmail.com

---

**[← Back to Main Course](../README.md)**
