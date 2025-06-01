# Titanic Dataset Analysis

## Overview

This project analyzes the **Titanic dataset** to explore patterns, trends, and relationships between passenger features and survival rates. The dataset includes details like passenger class, age, fare, and more. We use _Python_, _Pandas_, _Matplotlib_, and _Seaborn_ to load, summarize, and visualize the data.

## Dataset

The dataset (`Titanic-Dataset.csv`) contains passenger information from the Titanic, including:

- **PassengerId**: Unique ID for each passenger
- **Survived**: Survival status (0 = No, 1 = Yes)
- **Pclass**: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Age**: Passenger age
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Fare**: Ticket price

## Code Description

The Jupyter notebook (`day2.ipynb`) performs the following steps:

1. **Imports Libraries**: Uses _Pandas_ for data handling, _Matplotlib_ for plotting, and _Seaborn_ for enhanced visualizations.
2. **Loads Data**: Reads the Titanic dataset from a CSV file into a _Pandas_ DataFrame.
3. **Summarizes Data**: Displays statistical summaries (count, mean, std, min, max, etc.) for numeric columns.
4. **Identifies Numeric Columns**: Lists numeric features for analysis.
5. **Visualizes Relationships**: Creates a pairplot to show relationships between numeric features, colored by survival status.

## Key Findings

### Patterns and Trends

- **Survival by Class**: The pairplot shows higher survival rates for 1st-class passengers (`Pclass=1`) compared to 2nd and 3rd classes. Lower classes have more non-survivors.
- **Age and Survival**: Younger passengers (especially below 20) tend to have slightly higher survival rates, visible in the `Age` vs. `Survived` scatter and density plots.
- **Fare and Survival**: Higher fares correlate with better survival odds, likely tied to passenger class. Scatter plots of `Fare` vs. `Survived` show survivors paid more on average.
- **Family Size**: `SibSp` and `Parch` (siblings/spouses and parents/children) show small families (1-2 members) had better survival chances than larger groups or solo travelers.

### Anomalies

- **Missing Age Data**: The summary shows only 714 age values for 891 passengers, indicating missing data that could skew age-related insights.
- **Fare Outliers**: Fares range from 0 to 512.33, with some extremely high values. Free tickets (`Fare=0`) might indicate crew, children, or data errors.

### Feature-Level Inferences

- **`Pclass`**: Strong influence on survival—1st-class passengers likely had priority access to lifeboats.
- **`Age`**: Younger passengers may have been prioritized during evacuation, boosting survival rates.
- **`Fare`**: Higher fares, linked to better cabins and class, align with higher survival, suggesting socio-economic status played a role.
- **`SibSp` and `Parch`**: Small family sizes show better survival, possibly due to easier coordination during rescue.

## How to Use

1. **Requirements**: Install Python, then run:
   ```bash
   pip install pandas matplotlib seaborn
   ```
2. **Download Data**: Place `Titanic-Dataset.csv` in the same folder as the notebook.
3. **Run the Notebook**: Open `day2.ipynb` in _Jupyter_ and execute cells to load, summarize, and visualize the data.
4. **View Results**: Check the pairplot for visual insights and the summary for numeric details.
