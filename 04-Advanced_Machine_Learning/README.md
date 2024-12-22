# Module 4: Advanced Machine Learning

Welcome to Module 4 of the NeuroVision roadmap! In this module, we dive into advanced machine learning techniques and concepts to enhance model performance and tackle complex problems.

---

## **Topics Covered**

1. **Advanced Algorithms**
    - Decision Trees
    - Random Forests
    - Gradient Boosting Machines (XGBoost, LightGBM, CatBoost)
2. **Ensemble Methods**
    - Bagging
    - Boosting
    - Stacking
3. **Hyperparameter Tuning**
    - Grid Search
    - Random Search
    - Bayesian Optimization
4. **Model Validation and Cross-Validation**
    - K-Fold Cross-Validation
    - Stratified K-Fold
    - Leave-One-Out Cross-Validation (LOOCV)
5. **Practical Implementation**
    - Hands-on with Python libraries (Scikit-Learn, XGBoost, LightGBM)
    - Working with real datasets

---

## **1. Advanced Algorithms**

### Decision Trees:
- A non-parametric algorithm used for classification and regression.
- Creates a tree-like model of decisions based on feature values.
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)
```

### Random Forests:
- An ensemble of decision trees that reduces overfitting and improves accuracy.
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
```

### Gradient Boosting:
- Boosting technique that builds trees sequentially, reducing errors iteratively.
```python
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
```

---

## **2. Ensemble Methods**

### Bagging:
- Combines predictions from multiple models trained on different subsets of the data.
```python
from sklearn.ensemble import BaggingClassifier
model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10)
model.fit(X_train, y_train)
```

### Boosting:
- Combines weak learners to create a strong learner by focusing on misclassified examples.

### Stacking:
- Combines multiple models using a meta-model for final predictions.
```python
from sklearn.ensemble import StackingClassifier
model = StackingClassifier(estimators=[('rf', RandomForestClassifier()), ('gb', XGBClassifier())])
model.fit(X_train, y_train)
```

---

## **3. Hyperparameter Tuning**

### Grid Search:
- Exhaustively searches predefined hyperparameter space.
```python
from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth': [3, 5, 7]}
grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)
```

### Random Search:
- Randomly samples hyperparameters for faster tuning.
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_dist = {'max_depth': randint(1, 10)}
random_search = RandomizedSearchCV(DecisionTreeClassifier(), param_dist, cv=5)
random_search.fit(X_train, y_train)
```

### Bayesian Optimization:
- Uses probabilistic models to find optimal hyperparameters.

---

## **4. Model Validation and Cross-Validation**

### K-Fold Cross-Validation:
- Splits data into K subsets, trains on K-1, and tests on the remaining subset.
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
```

### Stratified K-Fold:
- Maintains class distribution across folds.
```python
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5)
```

### Leave-One-Out Cross-Validation (LOOCV):
- Uses one observation for validation and the rest for training.
```python
from sklearn.model_selection import LeaveOneOut
cv = LeaveOneOut()
```

---

## **5. Practical Implementation**

### Hands-On with Python Libraries:
- **Scikit-Learn**: Advanced algorithms and validation
- **XGBoost**: Gradient Boosting
- **LightGBM**: Fast Gradient Boosting

#### Example Workflow:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("data.csv")
X = data.drop('target', axis=1)
y = data['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")
```

### Working with Real Datasets:
- Explore datasets like Titanic, Housing Prices, or Credit Card Fraud on Kaggle.
- Focus on feature-rich, imbalanced datasets to apply advanced techniques.
