# 📊 Evaluation Metrics Quick Reference

## For Classification Models

---

## 🎯 Confusion Matrix Breakdown

```
                    Predicted
                   Neg    Pos
            ┌─────────┬─────────┐
Actual Neg  │   TN    │   FP    │
            ├─────────┼─────────┤
Actual Pos  │   FN    │   TP    │
            └─────────┴─────────┘
```

### Understanding Each Quadrant

| Cell | Name | Meaning | Example (Disease Detection) |
|------|------|---------|----------------------------|
| **TP** | True Positive | Correctly predicted positive | Detected disease when present ✅ |
| **TN** | True Negative | Correctly predicted negative | Cleared healthy patient ✅ |
| **FP** | False Positive | Wrongly predicted positive | False alarm (Type I error) ⚠️ |
| **FN** | False Negative | Wrongly predicted negative | Missed disease (Type II error) 🚨 |

---

## 📐 Core Metrics Formulas

### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
**Meaning**: Overall correct predictions
**Use when**: Classes are balanced
**Don't use when**: Imbalanced data (99% negative → always predict negative = 99% accuracy!)

### Precision
```
Precision = TP / (TP + FP)
```
**Meaning**: "Of all positive predictions, how many were correct?"
**Use when**: False positives are costly (spam filter - don't block good emails)

### Recall (Sensitivity, TPR)
```
Recall = TP / (TP + FN)
```
**Meaning**: "Of all actual positives, how many did we catch?"
**Use when**: False negatives are costly (disease screening - don't miss patients)

### F1 Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
**Meaning**: Harmonic mean of precision and recall
**Use when**: You need balance between precision and recall

### F-beta Score
```
Fβ = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
```
- **F0.5**: Weights precision higher (2× more important than recall)
- **F1**: Equal weight
- **F2**: Weights recall higher (2× more important than precision)

---

## 📈 ROC and AUC

### ROC Curve
Plots **True Positive Rate** vs **False Positive Rate** at various thresholds.

```
TPR (Recall) = TP / (TP + FN)
FPR = FP / (FP + TN)
```

### Interpreting ROC

| Curve Position | Quality |
|---------------|---------|
| Top-left corner (0, 1) | Perfect classifier |
| Close to top-left | Excellent model |
| Diagonal line | Random guess (useless) |
| Below diagonal | Worse than random |

### AUC (Area Under Curve)

| AUC Value | Interpretation |
|-----------|----------------|
| 1.0 | Perfect |
| 0.9 - 1.0 | Excellent |
| 0.8 - 0.9 | Good |
| 0.7 - 0.8 | Fair |
| 0.6 - 0.7 | Poor |
| 0.5 | Random |

**💡 Key Insight**: AUC = probability that model ranks random positive higher than random negative.

---

## ⚖️ Precision-Recall Trade-off

```
        High Precision              High Recall
             ↓                          ↓
    Few false alarms            Catch all positives
    But miss some cases         But more false alarms
```

### Precision-Recall Curve
Better for **imbalanced datasets** than ROC!

```
Average Precision (AP) = Area under PR curve
```

---

## 🎚️ Threshold Selection Guide

| Domain | Priority | Recommended Threshold | Reasoning |
|--------|----------|----------------------|-----------|
| Medical Screening | Recall | **0.2 - 0.4** | Don't miss diseases |
| Spam Detection | Precision | **0.7 - 0.9** | Don't block good emails |
| Fraud Detection | Recall | **0.3 - 0.5** | Catch all fraud |
| Credit Scoring | Balanced | **0.5 - 0.6** | Balance risk |
| Anomaly Detection | Precision | **0.8 - 0.95** | Minimize false alerts |

---

## 🎯 Multi-Class Metrics

### Micro Average
Pool all classes together, then calculate metric.
```
Micro-Precision = Total TP / (Total TP + Total FP)
```
**Good when**: All samples matter equally

### Macro Average
Calculate metric for each class, then average.
```
Macro-Precision = (P₁ + P₂ + ... + Pₖ) / K
```
**Good when**: All classes matter equally

### Weighted Average
Weight by class frequency.
```
Weighted-Precision = Σ (nᵢ × Pᵢ) / Σ nᵢ
```
**Good when**: More common classes should contribute more

---

## 📊 Quick Decision Tree

```
Which metric to use?
    │
    ├── Balanced classes?
    │   └── YES → Accuracy or F1
    │
    ├── Imbalanced classes?
    │   ├── FP is costly → Precision
    │   ├── FN is costly → Recall
    │   └── Need balance → F1, F2, or ROC-AUC
    │
    └── Multi-class?
        ├── All samples equal → Micro-Average
        ├── All classes equal → Macro-Average
        └── Weighted by frequency → Weighted-Average
```

---

## 💻 Python Code Snippets

### Classification Report
```python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
```

### ROC-AUC
```python
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_true, y_prob)
auc = roc_auc_score(y_true, y_prob)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
```

### Precision-Recall Curve
```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
ap = average_precision_score(y_true, y_prob)

plt.plot(recall, precision, label=f'AP = {ap:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
```

### Optimal Threshold for F1
```python
from sklearn.metrics import f1_score
import numpy as np

thresholds = np.arange(0.1, 1.0, 0.05)
f1_scores = [f1_score(y_true, (y_prob >= t).astype(int)) for t in thresholds]
optimal_threshold = thresholds[np.argmax(f1_scores)]
```

---

## ⚠️ Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Using accuracy on imbalanced data | Misleading results | Use F1, AUC, or PR-AUC |
| Ignoring threshold | Default 0.5 isn't optimal | Tune threshold for domain |
| Not using stratified split | Test set may be imbalanced | Use StratifiedKFold |
| Comparing AUC across datasets | AUC depends on class balance | Use same test set |
| Reporting only one metric | Incomplete picture | Report multiple metrics |

---

## 📚 Key Takeaways

> "Accuracy is the most dangerous metric - it can give you false confidence on imbalanced data."

> "In medicine, 99% precision with 1% recall is USELESS. You need to catch patients!"

> "Always ask: What's the COST of false positives vs false negatives?"

> "If you can only pick one metric, use ROC-AUC for balanced data or PR-AUC for imbalanced."

---

*Created for: Deep Dive into ML - Module 03 Logistic Regression*
