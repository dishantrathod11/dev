# dev
Customer Churn Prediction 
# Customer Churn Prediction Project

## 📅 Project Overview

This project aims to predict customer churn in a banking environment using machine learning. By analyzing customer demographics, account activity, and financial behavior, we develop a classification model that identifies customers likely to leave the bank.

## 👨‍💼 Business Objective

Reducing churn is critical for financial institutions, as acquiring new customers is far more costly than retaining existing ones. This model helps:

* Identify high-risk churn customers
* Enable proactive retention strategies
* Support business decisions with actionable insights

## 📊 Dataset

The dataset contains information on 10,000+ bank customers:

* Features: `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`, `Geography`, `Gender`, and `Exited` (target)
* Source: Simulated data based on a typical European bank

## 👁️ Exploratory Data Analysis (EDA)

* Identified class imbalance in churned vs non-churned customers
* Visualized feature distributions and relationships
* Detected trends with respect to age, balance, and account activity

## 💡 Feature Engineering

* Created interaction terms:

  * `Age_Tenure` = `Age * Tenure`
  * `Balance_Salary_Ratio` = `Balance / EstimatedSalary`
  * `CreditScore_Products` = `CreditScore * NumOfProducts`
* Encoded categorical variables (One-hot encoding for `Geography` and `Gender`)

## 🧹 Data Preprocessing

* Applied `StandardScaler` to normalize numerical features
* Handled class imbalance using **SMOTE (Synthetic Minority Oversampling Technique)**

## 🚀 Model Building

Tested and compared:

* Logistic Regression
* Random Forest Classifier

### Model Tuning:

Used `GridSearchCV` for hyperparameter tuning on the Random Forest model:

```python
param_grid = {
  'n_estimators': [100, 200],
  'max_depth': [4, 6, 10],
  'min_samples_split': [2, 5],
  'min_samples_leaf': [1, 2]
}
```

## 🔢 Model Evaluation

* Final Model: Tuned Random Forest with SMOTE
* Metrics:

  * Accuracy: 83%
  * F1-Score (Churn): 0.62
  * ROC AUC: 0.775
* Confusion Matrix:

```
[[1371  222]
 [ 126  281]]
```

## 🔍 Key Insights

* Churn is strongly influenced by:

  * Age
  * Account Activity (`IsActiveMember`)
  * Ratio of Balance to Salary
  * Tenure and Product combinations
* High churn likelihood in older, less active users with high balances

## 🔧 Deployment & Inference

* Saved model and scaler using `joblib`
* Designed prediction pipeline for manually entered customers:

```python
# Load model & scaler
model = joblib.load('tuned_random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Enter new customer data as DataFrame
# Apply same preprocessing & feature engineering
# Scale inputs and predict churn
```

## 🏆 Conclusion

This project demonstrates a full end-to-end data science workflow:

* From data exploration to predictive modeling
* Tackling class imbalance
* Engineering features to improve performance
* Providing business insights to reduce churn risk

## 👁️ Next Steps

* Try advanced models like XGBoost
* Use SHAP for explainable AI (model interpretability)
* Deploy as an API or web tool for business use

---

**Author:** Dishant Rathod
**Project Type:** Supervised Classification (Binary)
**Tools Used:** Python, Pandas, Seaborn, Scikit-learn, SMOTE, Matplotlib

> "Churn isn’t just a number. It's a customer who's one step from leaving. This model helps you know who and why."
