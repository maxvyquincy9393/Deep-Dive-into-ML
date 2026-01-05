# Module 07: Generalized Linear Models (GLM)

**Part of the Deep Dive Machine Learning Series** | Based on **Stanford CS229**

A unified framework connecting linear regression, logistic regression, and beyond through the exponential family.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![CS229](https://img.shields.io/badge/Stanford-CS229-red.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.20+-green.svg)

## Overview

Generalized Linear Models provide a beautiful unifying framework for many machine learning algorithms. By understanding the exponential family of distributions and link functions, you'll see how linear regression, logistic regression, and Poisson regression are all special cases of the same general framework.

## What You'll Learn

- **Exponential Family**: The general form $p(y;\eta) = b(y) \exp(\eta^T T(y) - a(\eta))$
- **Natural Parameter**: How η relates to the distribution parameters
- **Link Functions**: Connecting linear predictors to response means
- **Canonical Link**: Why logistic uses sigmoid, Poisson uses log
- **Poisson Regression**: Modeling count data
- **GLM Construction**: How to derive new models from distributions

## Notebooks

| Notebook | Description |
|----------|-------------|
| `Module_07_Generalized_Linear_Models.ipynb` | Complete GLM framework and implementations |

## Real-World Case Study

**Bike Rental Count Prediction** - Using Poisson regression to predict bike rental counts (count data with non-negative integers).

## Key Equations

**Exponential Family Form:**
$$p(y;\eta) = b(y) \exp(\eta^T T(y) - a(\eta))$$

**GLM Components:**
1. $y | x; \theta \sim$ ExponentialFamily($\eta$)
2. Goal: predict $E[T(y)|x]$
3. $\eta = \theta^T x$ (linear in inputs)

**Common GLMs:**

| Distribution | Link Function | Use Case |
|-------------|---------------|----------|
| Gaussian | Identity: $g(\mu) = \mu$ | Continuous outcomes |
| Bernoulli | Logit: $g(\mu) = \log\frac{\mu}{1-\mu}$ | Binary classification |
| Poisson | Log: $g(\mu) = \log(\mu)$ | Count data |
| Exponential | Inverse: $g(\mu) = 1/\mu$ | Waiting times |

## CS229 Connection

- **Lecture 4**: Generalized Linear Models
- **Lecture Notes**: Exponential Family and GLMs

## Key Insight

Linear Regression, Logistic Regression, and Softmax Regression are all GLMs:
- **Linear**: Gaussian distribution, identity link
- **Logistic**: Bernoulli distribution, logit link
- **Softmax**: Multinomial distribution, softmax link

## Related Modules

| Previous | Current | Next |
|----------|---------|------|
| [Module 06: Newton's Method](../Module_06_Newton_Perceptron/) | **Module 07: GLM** | [Module 08: GDA](../Module_08_GDA/) |

## Author

**Kanisius Bagas**

- GitHub: [@maxvyquincy9393](https://github.com/maxvyquincy9393)
- LinkedIn: [kanisiusbagas1212](https://www.linkedin.com/in/kanisiusbagas1212)
- Email: maxvy1218@gmail.com

---

**[← Back to Main Course](../README.md)**
