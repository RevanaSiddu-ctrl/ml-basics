import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


x = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 0, 1, 1, 1])

model = LogisticRegression()

scores = cross_val_score(model, x, y,  cv=3)

print(scores)
print(scores.mean())