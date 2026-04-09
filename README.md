# Logistic Regression with Cross Validation

### Overview

This project demonstrates how to use **Logistic Regression** with **Cross Validation** to evaluate model performance reliably.

The goal is to classify data into two categories (0 or 1) based on input features.

---

###  Concepts Covered

* Logistic Regression
* Model Training (`fit`)
* Prediction
* Cross Validation
* Accuracy Evaluation

---

### Dataset

Simple dataset used for understanding:

* **Input (X):** Number values representing a feature
* **Output (y):** Binary classification (0 or 1)

```python
X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]
```

---

### Implementation

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

x = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 0, 1, 1, 1])

model = LogisticRegression()

scores = cross_val_score(model, x, y, cv=3)

print(scores)
print(scores.mean())
```

---

### Cross Validation Explained

* Data is split into multiple parts (folds)
* Model is trained and tested multiple times
* Each time a different part is used as test data
* Final performance is the **average of all scores**

---

### Output

* `scores` → accuracy for each split
* `scores.mean()` → overall model performance

---

### Key Learnings

* Cross Validation provides more reliable results than a single train-test split
* Helps reduce dependency on random data splitting
* Improves confidence in model evaluation

---

###  Conclusion

This project shows how to:

* Train a classification model
* Evaluate it properly using cross validation
* Understand model performance stability

---

### Future Improvements

* Use larger datasets
* Add feature scaling
* Try different models (Decision Trees, Random Forest)

---

###Author

Beginner Machine Learning Project
