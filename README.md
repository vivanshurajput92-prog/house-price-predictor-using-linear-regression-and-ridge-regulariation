# 🏡 House Price Prediction: End-to-End Regression Pipeline

An automated Machine Learning pipeline built to predict continuous housing prices using tabular data. 

This project focuses on clean software architecture and mathematical optimization. Rather than manually cleaning individual columns, the core of this repository is a robust, automated Scikit-Learn `Pipeline` that scales, imputes, and encodes 79+ mixed data-type features without data leakage.

## ⚙️ The Engineering Architecture (Data Pipeline)
Real-world tabular data is messy. To ensure the models received a mathematically pure matrix, I built a `ColumnTransformer` to route features through isolated processing pipelines:

1. **Numerical Pipeline:** Handled missing continuous variables (like Square Footage) using a `Median` imputer, followed by a `StandardScaler` to normalize the distribution and prevent features with large absolute numbers from dominating the objective function.
2. **Categorical Pipeline:** Handled missing text categories using a `Constant` imputer, followed by `OneHotEncoder` to expand nominal strings (like Neighborhood or Roof Style) into a sparse binary matrix.

## 🧠 The Mathematical Progression

### Phase 1: The Dimensionality Problem (Linear Regression)
* **Approach:** The initial baseline model utilized standard Ordinary Least Squares (OLS) Linear Regression.
* **Result:** Achieved a massive Training $R^2$ score of `0.933`, but performed poorly on unseen test data (Kaggle RMSLE: `0.186`).
* **Diagnosis:** Overfitting due to the "Curse of Dimensionality." The One-Hot Encoder expanded the dataset from 79 columns into hundreds of binary features. The standard linear model became excessively greedy, assigning arbitrary weights to obscure categorical features and effectively memorizing the training set.

### Phase 2: L2 Regularization (Ridge Regression)
* **Approach:** To cure the high variance, I upgraded the predictive engine to **Ridge Regression**. 
* **Result:** The Kaggle RMSLE score improved drastically to **`0.149`**.
* **Diagnosis:** By injecting an L2 mathematical penalty ($\alpha$), the model was actively punished for relying on massive coefficients. This forced the algorithm to shrink the weights of noisy/useless features toward zero, resulting in a significantly smoother and more generalized decision boundary for unseen real estate data.

## 🚀 Technologies Used
* **Language:** Python 3
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (Pipelines, ColumnTransformer, Ridge, SimpleImputer, StandardScaler, OneHotEncoder)
* **Evaluation Metrics:** Root Mean Squared Logarithmic Error (RMSLE), Root Mean Squared Error (RMSE), $R^2$ Score
