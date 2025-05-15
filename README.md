# **Insurance Premium Prediction - Kaggle Competition**

## 🚀 **Machine Learning | LightGBM | Feature Engineering**

# **About**
This notebook documents my approach and learnings from participating in a Kaggle competition focused on predicting insurance premium amounts. The evaluation metric for this competition is **Root Mean Squared Logrithmic Error (RMSLE).** The dataset is split into three parts: train, test, and an original dataset:

* Train: 1.2 million records
* Test: 800,000 records
* Original: 278,860 records
  
## **Goal**
* The objective is to accurately predict the premium_amount for each policyholder, optimizing the model for **RMSLE.**

## **Challenges**
* **Missing Data:** Several features contain missing values, ranging from **2% to over 30%.**
* **Outliers:** Columns like **`annual_income`** and **`health_score`** contain extreme values. These values were retained as they likely reflect real-world variability across individuals.
* **Weak Correlation with Target Variable:** The dataset lacks strong linear relationships with the target variable. For instance, the highest observed correlation was just **0.047** between previous_claim and premium_amount, indicating minimal direct predictive power.

## **Approach**
To address these challenges and enhance model performance:

**Exploratory Data Analysis (EDA):**
Conducted in-depth analysis to understand feature distributions, identify outliers, and assess feature-target relationships.
Feature Engineering:

✅ Created interaction and ratio-based features to uncover deeper relationships between variables.

✅ Applied smoothed target encoding to categorical features and group-wise categorical combinations to enhance signal strength.

✅ For high-cardinality numerical features, performed quantile binning followed by target-based statistical aggregations.

**Handling Missing Values:**
* Applied median imputation for numerical features and mode imputation for categorical features using **SimpleImputer** within each fold via **ColumnTransformer,** ensuring **no data leakage** during cross-validation.
  
**Cross-Validation Strategy:**
* Employed **5-fold cross-validation.**
* All feature engineering steps including missing value imputation, target encoding, and aggregation were performed within each fold to prevent data leakage.

**Model**
* Implemented and trained a **LightGBM model** using the engineered feature set.
* Manually selected model hyperparameters through iterative experimentation and validation.
