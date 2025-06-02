# Linear Regression Project (Day 3)

This project demonstrates how to build a multiple linear regression model using Python and `scikit-learn`. It includes steps for preprocessing the data, splitting into training and testing sets, training the model, evaluating its performance, and attempting to visualize the results.

## Overview

### 1. Import and Preprocess the Dataset

- The dataset is loaded using `pandas`.
- Basic data cleaning is done: checking for missing values, dropping irrelevant columns, and preparing the features for training.
- Features are optionally scaled or encoded as needed.

### 2. Split the Data

- The dataset is split into training and testing sets using `train_test_split()` from `sklearn.model_selection`.
- This helps ensure that the model is evaluated fairly using data it hasn’t seen during training.

### 3. Fit a Linear Regression Model

- A linear regression model is created using `LinearRegression()` from `sklearn.linear_model`.
- The model learns the relationship between the input features and the target variable.

### 4. Evaluate the Model

- The model's performance is measured using:
  - **MAE (Mean Absolute Error)**
  - **MSE (Mean Squared Error)**
  - **R² Score**

### 5. Plot the Regression Line

- A regression line is not plotted in this project because the model uses **multiple features**.
- In simple linear regression (with one feature), the regression line is a 2D line that can be visualized easily.
- In this case, since the model uses multiple input variables, it generates a **hyperplane** in higher dimensions, which cannot be represented by a single line on a 2D plot.

---

## Requirements and Installation

Make sure you have the following Python packages installed:

- Python 3.7+
- numpy
- pandas
- matplotlib
- scikit-learn
- jupyter (if running the notebook)

You can install them with:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```
