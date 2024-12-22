# Module 3: Data Preprocessing and Feature Engineering

Welcome to Module 3 of the NeuroVision roadmap! In this module, we focus on the critical steps of preparing data for machine learning models. Data preprocessing and feature engineering form the backbone of successful machine learning pipelines, ensuring data quality and relevance for modeling.

---

## **Topics Covered**

1. **Data Cleaning**
    - Handling missing values
    - Dealing with outliers
    - Removing duplicates
    - Addressing data inconsistencies
2. **Feature Scaling and Normalization**
    - Standardization (Z-score scaling)
    - Min-Max scaling
    - Log transformation
3. **Feature Selection**
    - Statistical methods (Correlation, Chi-Square Test)
    - Recursive Feature Elimination (RFE)
    - LASSO Regularization
4. **Feature Extraction**
    - Principal Component Analysis (PCA)
    - Singular Value Decomposition (SVD)
    - Feature Encoding (One-Hot, Label Encoding)
5. **Practical Implementation**
    - Hands-on with Python libraries (Pandas, Scikit-Learn)
    - Working with real datasets

---

## **1. Data Cleaning**

Data cleaning involves identifying and correcting errors or inconsistencies in the data.

### Handling Missing Values:
- **Imputation:**
  - Mean/Median Imputation for numerical data
  - Mode Imputation for categorical data
  - Advanced techniques like K-Nearest Neighbors (KNN) Imputation
- **Removal:** Drop rows or columns with excessive missing data.

### Dealing with Outliers:
- Identify outliers using:
  - Z-score
  - Interquartile Range (IQR)
- Handle outliers by:
  - Capping and Flooring
  - Transforming data

### Removing Duplicates:
```python
import pandas as pd
# Drop duplicates
data = pd.DataFrame({"A": [1, 2, 2], "B": [3, 4, 4]})
data_cleaned = data.drop_duplicates()
```

### Addressing Data Inconsistencies:
- Uniform formatting (e.g., date formats, string capitalization)
- Standardizing units of measurement

---

## **2. Feature Scaling and Normalization**

Scaling and normalization ensure features are on a comparable scale, improving model performance.

### Standardization (Z-Score Scaling):
Converts data to have a mean of 0 and a standard deviation of 1.
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
```

### Min-Max Scaling:
Scales data to a specific range, often [0, 1].
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
```

### Log Transformation:
Useful for skewed data.
```python
import numpy as np
data_transformed = np.log1p(data)
```

---

## **3. Feature Selection**

Feature selection helps identify the most relevant features, improving model interpretability and performance.

### Statistical Methods:
- **Correlation Matrix:** Identify highly correlated features.
- **Chi-Square Test:** Assess the relationship between categorical features.

### Recursive Feature Elimination (RFE):
Iteratively removes less important features.
```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
rfe = RFE(RandomForestClassifier(), n_features_to_select=5)
X_selected = rfe.fit_transform(X, y)
```

### LASSO Regularization:
Penalizes less important features using L1 regularization.
```python
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
important_features = lasso.coef_ != 0
```

---

## **4. Feature Extraction**

Feature extraction transforms raw data into meaningful features.

### Principal Component Analysis (PCA):
Reduces dimensionality while preserving variance.
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
```

### Singular Value Decomposition (SVD):
Another method for dimensionality reduction.

### Feature Encoding:
- **One-Hot Encoding:** Converts categorical variables into binary vectors.
  ```python
  pd.get_dummies(data['category'])
  ```
- **Label Encoding:** Assigns numerical labels to categories.
  ```python
  from sklearn.preprocessing import LabelEncoder
  le = LabelEncoder()
  data['encoded'] = le.fit_transform(data['category'])
  ```

---

## **5. Practical Implementation**

### Hands-On with Python Libraries:
- **Pandas:** Data manipulation and cleaning
- **Scikit-Learn:** Scaling, selection, and extraction

#### Example Workflow:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load dataset
data = pd.read_csv("data.csv")

# Handling missing values
imputer = SimpleImputer(strategy="mean")
data_cleaned = imputer.fit_transform(data)

# Feature scaling
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_cleaned)

# Encoding categorical features
encoder = OneHotEncoder()
data_encoded = encoder.fit_transform(data[['category']])
```

### Working with Real Datasets:
- Explore Kaggle datasets for hands-on practice.
- Focus on datasets like Titanic Survival, Housing Prices, or Credit Card Fraud.
