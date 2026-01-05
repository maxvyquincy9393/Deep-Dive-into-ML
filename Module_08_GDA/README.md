# Module 08: Gaussian Discriminant Analysis (GDA)

**Part of the Deep Dive Machine Learning Series** | Based on **Stanford CS229**

A comprehensive implementation and analysis of Gaussian Discriminant Analysis, exploring the mathematical foundations, practical applications, and real-world performance of generative classification models.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![CS229](https://img.shields.io/badge/Stanford-CS229-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.20+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> *"GDA is a generative learning algorithm that models P(x|y) - it asks 'what does a typical class 0 example look like?' rather than 'where is the boundary between classes?'"* - CS229 Lecture Notes

## Overview

This project provides an in-depth exploration of **Gaussian Discriminant Analysis (GDA)**, a powerful generative classification algorithm. Unlike discriminative models that learn decision boundaries directly, GDA models the underlying probability distributions of each class using multivariate Gaussians.

### Key Features

- **From-scratch implementation** of GDA with Maximum Likelihood Estimation (MLE)
- **Mathematical derivations** of all parameter estimates (φ, μ, Σ)
- **Comparison study**: GDA vs Logistic Regression under different data conditions
- **LDA vs QDA analysis**: When to use linear vs quadratic discriminant analysis
- **Assumption violation testing**: Demonstrating GDA failure modes on non-Gaussian data
- **Real-world case study**: Iris dataset classification with performance benchmarking

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Key Concepts](#key-concepts)
- [Results](#results)
- [Usage](#usage)
- [Mathematical Foundation](#mathematical-foundation)
- [When to Use GDA](#when-to-use-gda)
- [Contributing](#contributing)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gda-deep-dive.git
cd gda-deep-dive

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
scipy>=1.7.0
jupyter>=1.0.0
```

## Project Structure

```
Module_08_GDA/
├── Module_08_GDA.ipynb          # Main notebook with complete implementation
├── README.md                     # This file
├── Real_Case_Study/
│   └── ...                       # Additional case studies
└── requirements.txt              # Python dependencies
```

## Key Concepts

### Generative vs Discriminative Models

| Aspect | GDA (Generative) | Logistic Regression (Discriminative) |
|--------|------------------|--------------------------------------|
| **Models** | P(x\|y) and P(y) | P(y\|x) directly |
| **Approach** | Learns class distributions | Learns decision boundary |
| **Sample Efficiency** | High (20-50 samples/class) | Lower (100+ samples/class) |
| **Assumptions** | Gaussian class-conditionals | None on P(x\|y) |
| **Interpretability** | Explicit class parameters | Boundary coefficients only |

### LDA vs QDA

- **Linear Discriminant Analysis (LDA)**: Assumes shared covariance matrix across all classes → Linear decision boundaries
- **Quadratic Discriminant Analysis (QDA)**: Allows separate covariance per class → Curved decision boundaries

## Results

### Performance Comparison on Gaussian Data

| Model | Accuracy | Training Time |
|-------|----------|---------------|
| GDA (LDA) | 98.0% | ~1 ms |
| Logistic Regression | 97.8% | ~15 ms |

**Key Finding**: On Gaussian-distributed data, GDA matches or exceeds Logistic Regression accuracy while training 10-15x faster due to closed-form MLE solutions.

### Sample Efficiency Analysis

```
Training Samples    GDA Accuracy    LogReg Accuracy
        20              ~92%            ~78%
        50              ~95%            ~88%
       100              ~97%            ~95%
       500              ~98%            ~98%
```

**Key Finding**: GDA reaches 95% accuracy with 50 samples per class, while Logistic Regression needs 100+ samples for comparable performance.

### Gaussian Assumption Violation

When tested on exponentially-distributed (non-Gaussian) data:
- GDA accuracy drops significantly
- Logistic Regression maintains robust performance

**Lesson**: Always verify distributional assumptions before using GDA.

## Usage

### Quick Start

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Train GDA (LDA)
gda = LinearDiscriminantAnalysis()
gda.fit(X_train, y_train)

# Predict
predictions = gda.predict(X_test)

# Access learned parameters
print(f"Class priors: {gda.priors_}")
print(f"Class means: {gda.means_}")
print(f"Shared covariance: {gda.covariance_}")
```

### From-Scratch Implementation

```python
import numpy as np

def fit_gda(X, y):
    """
    Fit Gaussian Discriminant Analysis using MLE.
    
    Returns:
        phi: Class prior P(y=1)
        mu_0, mu_1: Class means
        sigma: Shared covariance matrix
    """
    n = len(y)
    
    # MLE for prior
    phi = np.mean(y == 1)
    
    # MLE for class means
    mu_0 = X[y == 0].mean(axis=0)
    mu_1 = X[y == 1].mean(axis=0)
    
    # MLE for shared covariance
    X_centered = X.copy()
    X_centered[y == 0] -= mu_0
    X_centered[y == 1] -= mu_1
    sigma = (X_centered.T @ X_centered) / n
    
    return phi, mu_0, mu_1, sigma
```

## Mathematical Foundation

### Maximum Likelihood Estimation

GDA learns parameters by maximizing the log-likelihood:

$$\ell(\phi, \mu_0, \mu_1, \Sigma) = \sum_{i=1}^m \left[ \log P(x^{(i)} | y^{(i)}) + \log P(y^{(i)}) \right]$$

### Closed-Form Solutions

**Prior (φ):**
$$\phi = \frac{1}{m} \sum_{i=1}^m \mathbf{1}\{y^{(i)} = 1\}$$

**Class Means (μⱼ):**
$$\mu_j = \frac{\sum_{i: y^{(i)}=j} x^{(i)}}{\sum_{i: y^{(i)}=j} 1}$$

**Shared Covariance (Σ):**
$$\Sigma = \frac{1}{m} \sum_{i=1}^m (x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T$$

### Prediction via Bayes' Rule

$$P(y = k | x) = \frac{P(x | y = k) \cdot P(y = k)}{\sum_j P(x | y = j) \cdot P(y = j)}$$

## When to Use GDA

### Use GDA When:
- **Small datasets** (20-100 samples per class)
- **Continuous, approximately Gaussian features** (biomarkers, physical measurements)
- **Interpretability is important** (scientific/medical applications)
- **Outlier detection is needed** (fraud detection, anomaly detection)

### Avoid GDA When:
- **Large datasets available** (1000+ samples) → Use Logistic Regression
- **Non-Gaussian data** (count data, categorical, heavy-tailed) → Use tree-based methods
- **Complex decision boundaries needed** → Use neural networks or ensemble methods

## Visualizations

The notebook includes comprehensive visualizations:

1. **Gaussian Distribution Contours**: Topographic view of class distributions
2. **Decision Boundary Comparison**: LDA vs QDA boundary geometry
3. **Sample Efficiency Curves**: GDA vs Logistic Regression learning curves
4. **Confusion Matrix Analysis**: Error diagnosis for Iris classification
5. **Assumption Violation Demo**: Performance degradation on non-Gaussian data

## Key Takeaways

1. **Generative vs Discriminative**: GDA models how data is generated, not just boundaries
2. **Sample Efficiency**: GDA excels with limited data when Gaussian assumption holds
3. **Closed-Form Solutions**: No iterations needed - compute means, covariance, done
4. **Assumption Sensitivity**: Performance degrades significantly on non-Gaussian data
5. **LDA vs QDA Tradeoff**: More flexibility (QDA) requires more data to avoid overfitting

## Technologies Used

- **Python 3.8+**: Core programming language
- **NumPy**: Numerical computations and linear algebra
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms and utilities
- **SciPy**: Statistical functions (multivariate normal)
- **Jupyter Notebook**: Interactive development environment

## Author

**[Your Name]**

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## CS229 Connection

This module implements concepts from **Stanford CS229: Machine Learning** (Andrew Ng):

### Lecture Coverage
- **Lecture 5**: Generative Learning Algorithms
- **Lecture Notes**: [Generative Learning Algorithms](http://cs229.stanford.edu/notes2022fall/main_notes.pdf) (Section 4)

### Key CS229 Concepts Implemented
1. **Generative vs Discriminative Models** (CS229 Lecture 5)
   - GDA models P(x|y) × P(y), then uses Bayes' Rule for P(y|x)
   - Logistic Regression models P(y|x) directly

2. **Maximum Likelihood Estimation** (CS229 Lecture 2-3)
   - Closed-form solutions for φ, μ₀, μ₁, Σ
   - No iterative optimization required

3. **Gaussian Distribution Properties** (CS229 Lecture 5)
   - Multivariate Gaussian: $\mathcal{N}(\mu, \Sigma)$
   - Contours as ellipses determined by covariance

4. **LDA vs QDA Tradeoff** (CS229 Lecture 5)
   - Shared vs separate covariance matrices
   - Linear vs quadratic decision boundaries

### How This Extends CS229
While CS229 provides the theory, this notebook adds:
- **Visualization**: See the Gaussian "mountains" and decision boundaries
- **Comparison Studies**: Empirical GDA vs LogReg across sample sizes
- **Assumption Testing**: What happens when Gaussian assumption fails
- **Real-World Application**: Iris dataset classification benchmark

## Related Modules

| Previous | Current | Next |
|----------|---------|------|
| [Module 07: GLM](../Module_07_GLM/) | **Module 08: GDA** | [Module 09: Naive Bayes](../Module_09_Naive_Bayes/) |

### Prerequisites
- Module 03: Logistic Regression (discriminative baseline)
- Basic probability theory (Bayes' Rule, conditional probability)
- Linear algebra (matrix operations, covariance)

### What's Next
- **Module 09: Naive Bayes** - Another generative model with conditional independence assumption
- **Module 10: SVM** - Back to discriminative, but with margin maximization

## Acknowledgments

- **Stanford CS229** Machine Learning course by Andrew Ng
- CS229 Teaching Staff for comprehensive lecture notes

## Author

**Kanisius Bagas**

Part of the [Deep Dive Machine Learning](../) portfolio series.

- GitHub: [@maxvyquincy9393](https://github.com/maxvyquincy9393)
- LinkedIn: [kanisiusbagas1212](https://www.linkedin.com/in/kanisiusbagas1212)
- Email: maxvy1218@gmail.com

---

**[← Back to Main Portfolio](../README.md)**

**If you found this helpful, please give the repo a ⭐!**
