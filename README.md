# Deep Dive Machine Learning: From Theory to Implementation

<div align="center">

![ML Banner](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Deep%20Dive%20ML&fontSize=60&fontColor=fff&animation=twinkling&fontAlignY=35&desc=From%20Theory%20to%20Implementation&descAlignY=55&descSize=20)

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

[![CS229](https://img.shields.io/badge/Based%20on-Stanford%20CS229-8C1515?style=flat-square)](http://cs229.stanford.edu/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Modules](https://img.shields.io/badge/Modules-8%20Complete-blue?style=flat-square)]()

</div>

---

## About This Project

A comprehensive machine learning curriculum implementing every algorithm **from scratch** with mathematical rigor. Built while studying **Stanford CS229** by Andrew Ng.

<table>
<tr>
<td width="50%">

### What's Inside
- Mathematical derivations from first principles
- Pure NumPy implementations (no black boxes)
- Comparison with scikit-learn
- Real-world case studies
- Interactive visualizations

</td>
<td width="50%">

### Why From Scratch?
> *"The goal is not just to use machine learning, but to understand it deeply enough to invent new algorithms."*
> 
> — Andrew Ng

</td>
</tr>
</table>

---

## Curriculum

| # | Module | Topics | Case Study |
|:-:|--------|--------|------------|
| 01 | **[Linear Regression](Module_01_Linear_Regression/)** | Gradient Descent, Normal Equation | California Housing |
| 02 | **[Advanced Linear Regression](Module_02_Advanced_Linear_Regression/)** | Ridge, Lasso, Elastic Net | Medical Cost |
| 03 | **[Logistic Regression](Module_03_Logistic_Regression/)** | Binary Classification, Cross-Entropy | Heart Disease |
| 04 | **[Softmax Regression](Module_04_Softmax_Regression/)** | Multiclass, OvR vs OvO | Wine Quality |
| 05 | **[LWLR](Module_05_LWLR/)** | Non-parametric, Bandwidth | Temperature |
| 06 | **[Newton & Perceptron](Module_06_Newton_Perceptron/)** | 2nd Order Optimization | Binary Comparison |
| 07 | **[GLM](Module_07_GLM/)** | Exponential Family, Poisson, Gamma | Bike Rental |
| 08 | **[GDA](Module_08_GDA/)** | Generative Models, LDA/QDA | Iris Species |

---

## Sample Code

Every algorithm implemented from scratch:

```python
# Logistic Regression - Pure NumPy
def fit(X, y, lr=0.01, epochs=1000):
    theta = np.zeros(X.shape[1])
    
    for _ in range(epochs):
        h = 1 / (1 + np.exp(-X @ theta))  # Sigmoid
        gradient = X.T @ (h - y) / len(y)
        theta -= lr * gradient
    
    return theta
```

---

## Quick Start

```bash
git clone https://github.com/maxvyquincy9393/Deep-Dive-into-ML.git
cd Deep-Dive-into-ML
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook
```

---

## Tech Stack

<div align="center">

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **Core** | NumPy, Pandas, SciPy |
| **Visualization** | Matplotlib, Seaborn |
| **ML Reference** | scikit-learn |
| **Environment** | Jupyter Notebook |

</div>

---

## References

- [Stanford CS229](http://cs229.stanford.edu/) - Machine Learning by Andrew Ng
- [CS229 Lecture Notes](http://cs229.stanford.edu/notes2022fall/)
- The Elements of Statistical Learning - Hastie, Tibshirani, Friedman

---

## Author

<div align="center">

**Kanisius Bagas**

[![GitHub](https://img.shields.io/badge/GitHub-maxvyquincy9393-181717?style=for-the-badge&logo=github)](https://github.com/maxvyquincy9393)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-kanisiusbagas1212-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/kanisiusbagas1212)
[![Email](https://img.shields.io/badge/Email-maxvy1218%40gmail.com-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:maxvy1218@gmail.com)

</div>

---

<div align="center">

**If this helped your ML journey, give it a ⭐**

![Footer](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer)

</div>
