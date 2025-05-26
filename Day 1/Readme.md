# Titanic Dataset Analysis

This project analyzes the Titanic dataset to prepare it for machine learning tasks. The code is written in Python using a Jupyter notebook (`task1.ipynb`). Below is a summary of what the notebook does.

## Overview

The goal is to clean and preprocess the Titanic dataset to make it ready for building a predictive model (like predicting survival). The dataset includes passenger details like age, sex, ticket class, and survival status.

## Steps in the Analysis

1. **Loading the Data**

   - The dataset (`Titanic-Dataset.csv`) is loaded using `pandas`.
   - Libraries used: `pandas`, `numpy`, `matplotlib`, `seaborn` and `scikit-learn`.

2. **Exploring the Data**

   - The `.head()` and `.tail()` method is used to check the first five and last five rows of the dataset.
   - Boxplots are created for numerical columns (like `Age`, `Fare`, etc.) to visualize outliers.

3. **Handling Outliers**

   - Numerical columns are identified using `select_dtypes`.
   - A custom function `handle_outliers` is used to cap outliers (values outside 1.5 \* IQR) instead of removing them.
   - This ensures extreme values don’t skew the analysis.

4. **Encoding Categorical Data**

   - Categorical columns (like `Sex`) are identified.
   - Binary columns (with exactly two unique values) are encoded as 0 and 1 using a mapping (e.g., `male` → 0, `female` → 1).
   - A commented-out section suggests one-hot encoding for multi-category columns (like `Embarked`), but it’s not implemented in the code.

5. **Normalizing Numerical Data**
   - Numerical columns are standardized using `StandardScaler` from `sklearn` to scale them to a mean of 0 and standard deviation of 1.
   - This makes the data more suitable for machine learning models.

## Requirements

To run the notebook, you need:

- Python 3.x
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

Install them using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
