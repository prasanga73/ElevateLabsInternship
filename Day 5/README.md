# Heart Disease Prediction Project

## Overview

This project implements a machine learning analysis to predict heart disease using the `heart.csv` dataset within the `day5.ipynb` Jupyter Notebook. The following objectives were addressed:

1. **Trained a Decision Tree Classifier and Visualized It**  
   A Decision Tree Classifier was trained to predict heart disease based on features such as age, sex, chest pain type (`cp`), and others. However, the notebook lacks code to visualize the tree structure. This can be achieved by adding `sklearn.tree.plot_tree` to display the decision-making process.
2. **Analyzed Overfitting and Controlled Tree Depth**  
   A validation curve was implemented using `sklearn.model_selection.validation_curve` to analyze the effect of `max_depth` on training and validation performance. This helps identify optimal tree depth to control overfitting, though the notebook does not explicitly apply the findings to tune the Decision Tree Classifier.
3. **Trained a Random Forest and Compared Accuracy**  
   Currently, only a Decision Tree Classifier is implemented. Adding a `RandomForestClassifier` from `sklearn.ensemble` would enable training an ensemble model, with accuracy compared to the decision tree using `accuracy_score` from `sklearn.metrics`.
4. **Interpreted Feature Importances**  
   Feature importances were computed for the Decision Tree Classifier and visualized in a bar chart using `matplotlib`. Key predictors, such as chest pain type (`cp`), maximum heart rate (`thalach`), and others, were identified as influential in heart disease prediction.
5. **Evaluated Using Cross-Validation**  
   Cross-validation was conducted with `cross_val_score` (5 folds) for two models, referred to as `model` and `model1`. Results show mean accuracies of 1.0 and 0.997, respectively, with standard deviations of 0.0 and 0.0059, indicating potential stability but raising concerns about overfitting or data leakage for `model`.

## Usage Instructions

1. **Requirements**
   - Obtain the `day5.ipynb` Jupyter Notebook and the `heart.csv` dataset.
   - Ensure Python is installed, along with required libraries: `pandas`, `scikit-learn`, and `matplotlib`.
2. **Execution**
   - Open `day5.ipynb` in Jupyter Notebook.
   - Execute cells sequentially to load data, preprocess, train models, and view results.
3. **Outputs**
   - A preview of the dataset (`df.head()`) with columns: `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, and `target`.
   - A bar chart of feature importances for the Decision Tree Classifier.
   - Cross-validation scores, mean accuracy, and standard deviation for two models.
   - A validation curve plotting training and validation scores against `max_depth`.

## Tools and Technologies

- **Python Libraries**:
  - `pandas`: For loading and manipulating the `heart.csv` dataset.
  - `scikit-learn`: For preprocessing (`StandardScaler`, `train_test_split`), model training (`DecisionTreeClassifier`), evaluation (`accuracy_score`, `cross_val_score`), and validation curve analysis (`validation_curve`).
  - `matplotlib`: For plotting feature importances and the validation curve.
- **Dataset**: The `heart.csv` file, containing 14 columns of health-related features and a binary `target` column (0 = no heart disease, 1 = heart disease).

## Notes and Improvements

- **Overfitting Analysis**: A validation curve for `max_depth` is present, but the findings are not applied to tune the Decision Tree Classifier. Use the optimal `max_depth` from the curve to improve model generalization.
- **Data Quality**: No missing values were detected (`df.isnull().sum()`), but the perfect cross-validation score (1.0) for `model` suggests potential overfitting or data leakage.
