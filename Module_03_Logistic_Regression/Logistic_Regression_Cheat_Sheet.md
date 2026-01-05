# 📋 Logistic Regression Cheat Sheet

## Quick Reference Guide for Classification

---

## 🎯 When to Use Logistic Regression

| ✅ Use It | ❌ Avoid It |
|----------|------------|
| Binary classification (yes/no) | Multi-class (>10 classes) |
| Need probability outputs | Complex non-linear boundaries |
| Interpretable coefficients needed | Images, audio, text |
| Linearly separable data | Small datasets (<50 samples) |
| Fast training required | Highly imbalanced data |

---

## 📐 Core Formulas

### Sigmoid Function (Logistic Function)
```
σ(z) = 1 / (1 + e^(-z))
```

Maps any real number to [0, 1] - perfect for probability!

### Hypothesis (Prediction)
```
h_θ(x) = σ(θᵀx) = 1 / (1 + e^(-θᵀx))
```

Output: P(y=1 | x; θ) - probability of positive class

### Decision Rule
```
ŷ = 1  if h_θ(x) ≥ 0.5  (default threshold)
ŷ = 0  otherwise
```

### Cost Function (Cross-Entropy / Log Loss)
```
J(θ) = -1/m Σ [y⁽ⁱ⁾ log(h_θ(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾) log(1-h_θ(x⁽ⁱ⁾))]
```

Why log? Penalizes confident wrong predictions heavily!

### Gradient (for Gradient Descent)
```
∂J/∂θⱼ = 1/m Σ (h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾) · xⱼ⁽ⁱ⁾
```

Same form as linear regression - elegant!


### Update Rule
```
θⱼ := θⱼ - α · ∂J/∂θⱼ
```

---

## 📊 Evaluation Metrics

### Confusion Matrix
```
                    Predicted
                    0       1
Actual  0          TN      FP
        1          FN      TP
```

### Key Metrics

| Metric | Formula | Use When |
|--------|---------|----------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Balanced classes |
| **Precision** | TP/(TP+FP) | Minimize false positives |
| **Recall** | TP/(TP+FN) | Minimize false negatives |
| **F1 Score** | 2·(P·R)/(P+R) | Balance P and R |
| **ROC-AUC** | Area under ROC curve | Overall performance |

### When to Prioritize What

| Domain | Priority | Reason |
|--------|----------|--------|
| Medical diagnosis | **Recall** | Don't miss sick patients |
| Spam detection | **Precision** | Don't block good emails |
| Fraud detection | **Recall** | Catch all fraudsters |
| Customer churn | **F1 Score** | Balance matters |

---

## 🔧 Regularization

### L2 Regularization (Ridge)
```
J(θ) = -1/m Σ [cost] + λ/(2m) Σ θⱼ²
```
- Shrinks coefficients toward zero
- Never exactly zero
- Good when all features matter

### L1 Regularization (Lasso)
```
J(θ) = -1/m Σ [cost] + λ/m Σ |θⱼ|
```
- Can zero out coefficients
- Automatic feature selection
- Good for sparse models

### Elastic Net (L1 + L2)
```
J(θ) = -1/m Σ [cost] + λ₁ Σ |θⱼ| + λ₂ Σ θⱼ²
```
- Best of both worlds

---

## 🎚️ Threshold Tuning

Default threshold = 0.5 is NOT always optimal!

### How to Choose Threshold

```python
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

# Find optimal F1 threshold
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_threshold = thresholds[np.argmax(f1_scores)]
```

### Domain-Specific Thresholds

| Domain | Recommended Threshold | Reasoning |
|--------|----------------------|-----------|
| Medical screening | **0.3-0.4** | Lower → catch more cases |
| Spam filter | **0.7-0.8** | Higher → fewer false positives |
| Fraud detection | **0.2-0.3** | Lower → catch more fraud |
| Credit scoring | **0.5-0.6** | Balanced decision |

---

## 💻 Quick Code Snippets

### From Scratch Implementation
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_proba(X, theta):
    return sigmoid(X @ theta)

def cross_entropy_loss(y, h):
    return -np.mean(y * np.log(h + 1e-15) + (1-y) * np.log(1-h + 1e-15))

def gradient(X, y, h):
    return X.T @ (h - y) / len(y)
```

### sklearn Implementation
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Train
model = LogisticRegression(C=1.0, penalty='l2', max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
```

### Complete Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(C=0.1))
])

# Cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
print(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Train final model
pipeline.fit(X_train, y_train)
```

---

## ⚠️ Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Not scaling features | Always StandardScaler before Logistic Reg |
| Using accuracy on imbalanced data | Use F1, ROC-AUC, or Precision/Recall |
| Default threshold = 0.5 | Tune threshold based on domain |
| Ignoring convergence warnings | Increase max_iter or reduce C |
| Feature correlation | Check VIF, consider regularization |

---

## 🔄 Multi-Class Extensions

### One-vs-Rest (OvR)
- Train K binary classifiers
- Each: "class k vs all others"
- Prediction: class with highest probability

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(multi_class='ovr')
```

### Softmax (Multinomial)
- Single model, K outputs
- Uses softmax instead of sigmoid
- More principled for multi-class

```python
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
```

---

## 📈 Interpreting Coefficients

### Coefficient Meaning
```
log(p/(1-p)) = θ₀ + θ₁x₁ + θ₂x₂ + ...
```

- θⱼ > 0: Feature increases probability of class 1
- θⱼ < 0: Feature decreases probability of class 1
- |θⱼ| large: Feature has strong influence

### Odds Ratio
```
For 1-unit increase in xⱼ:
Odds multiply by e^θⱼ
```

Example: θ_age = 0.05 → each year adds 5% to odds

---

## 🧪 Model Diagnostics

### Check These Before Deploying

1. **Learning Curve**: Training vs validation error across data sizes
2. **Calibration Curve**: Predicted probability vs actual frequency
3. **Feature Importance**: |coefficient| bar chart
4. **Residual Analysis**: Deviance residuals
5. **ROC Curve**: Trade-off visualization

### Good Model Indicators

- ✅ ROC-AUC > 0.7 (good), > 0.8 (excellent)
- ✅ Calibration curve close to diagonal
- ✅ Small gap between train and validation error
- ✅ Consistent F1 across cross-validation folds

---

## 📚 Key Insights

> "Logistic regression is the go-to baseline for classification. It's simple, interpretable, and often surprisingly competitive with complex models."

> "The sigmoid function is the key insight - it squashes any input to [0,1], giving us a probability interpretation."

> "Cross-entropy loss is derived from Maximum Likelihood Estimation. It's not arbitrary - it's statistically principled."

> "In medicine, a model with 80% accuracy but 50% recall for disease is DANGEROUS. Always tune for domain-appropriate metrics."

---

## 🚀 Production Checklist

- [ ] Features are scaled (StandardScaler)
- [ ] Cross-validation performed (5-10 folds)
- [ ] Threshold tuned for domain
- [ ] Class imbalance addressed (if any)
- [ ] Calibration checked (for probability outputs)
- [ ] Feature importance reviewed
- [ ] Model saved with joblib
- [ ] Monitoring strategy defined

---

*Created for: Deep Dive into ML - Module 03 Logistic Regression*
