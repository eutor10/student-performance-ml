# Predicting Student Final Academic Performance Using Machine Learning

## Project Overview

This project investigates whether student demographic, academic, family, social, and lifestyle characteristics can be used to predict students' final academic grades using machine learning.

The project uses a student-performance dataset containing **395 students and 33 variables**. The final grade, represented by `G3`, is treated as the target variable.

A key design decision was to exclude `G1` and `G2`, which represent earlier-period grades, from the predictor variables. This was done to avoid data leakage and to evaluate whether final academic performance can be predicted using other available student characteristics.

Three regression models were developed and compared:

* Linear Regression
* Random Forest Regression
* Gradient Boosting Regression

Random Forest produced the strongest overall performance on the primary test split.

---

## Research Question

**Can student demographic, academic, family, social, and lifestyle characteristics be used to predict students' final academic grades (`G3`) using machine learning?**

---

## Project Objectives

The objectives of this project were to:

* Explore patterns and relationships within the student-performance dataset.
* Prepare numerical and categorical variables for machine learning.
* Predict students' final grades using regression models.
* Compare Linear Regression, Random Forest, and Gradient Boosting.
* Evaluate models using MAE, RMSE, and R².
* Tune the Random Forest model using cross-validation.
* Evaluate model stability across different train/test splits.
* Identify important predictors using feature-importance techniques.
* Analyze the model's largest prediction errors.
* Conduct a feature-ablation experiment to evaluate the contribution of important predictors.

---

## Dataset

The dataset contains:

* **395 rows**
* **33 columns**

The target variable is:

```text
G3
```

`G3` represents the student's final grade.

### Predictor Variables

After removing `G1`, `G2`, and `G3`, the project uses **30 predictor variables**.

### Numerical Variables

```text
age
Medu
Fedu
traveltime
studytime
failures
famrel
freetime
goout
Dalc
Walc
health
absences
```

### Categorical Variables

```text
school
sex
address
famsize
Pstatus
Mjob
Fjob
reason
guardian
schoolsup
famsup
paid
activities
nursery
higher
internet
romantic
```

---

## Data Preprocessing

The project uses a preprocessing pipeline to handle the different types of variables.

### Numerical Features

Numerical variables are processed using standardization where required by the model.

### Categorical Features

Categorical variables are transformed using one-hot encoding.

The preprocessing pipeline combines these transformations using a `ColumnTransformer`.

This approach allows numerical and categorical variables to be processed consistently before being passed to the machine-learning models.

---

## Preventing Data Leakage

`G1` and `G2` were deliberately removed from the predictor variables:

```python
X = df.drop(columns=["G1", "G2", "G3"])
y = df["G3"]
```

This prevents the models from directly using previous-period grades to predict the final grade.

The approach makes the research question more meaningful by focusing on other student characteristics rather than relying on previous grades.

---

## Train/Test Split

The dataset was divided using an 80/20 train/test split with:

```text
random_state = 42
```

The resulting datasets were:

```text
Training features: (316, 30)
Testing features:  (79, 30)

Training target: (316,)
Testing target:  (79,)
```

The test set was kept separate for evaluating model performance on unseen observations.

---

# Exploratory Data Analysis

Several exploratory analyses were conducted before model development.

## Final Grade Distribution

The most frequent final grade was **10**, with more than 50 students receiving that grade.

Most observations were concentrated approximately between grades **8 and 15**, although extreme grades were also present.

## Previous Failures

Previous failures showed one of the strongest relationships with final grade.

| Previous Failures | Mean G3 |
| ----------------: | ------: |
|                 0 |   11.25 |
|                 1 |    8.12 |
|                 2 |    6.24 |
|                 3 |    5.69 |

The correlation between `failures` and `G3` was:

```text
-0.360
```

This indicates a strong negative association relative to the other numerical predictors examined.

## Absences

The correlation between `absences` and `G3` was only:

```text
0.034
```

However, absences later became one of the most important predictors according to the machine-learning feature-importance analysis.

This demonstrates that weak linear correlation does not necessarily mean that a variable has little predictive value for a nonlinear model.

## Higher Education Aspiration

Students who indicated that they wanted to pursue higher education had a higher average final grade:

| Higher Education Aspiration | Count | Mean G3 |
| --------------------------- | ----: | ------: |
| No                          |    20 |    6.80 |
| Yes                         |   375 |   10.61 |

This result is interpreted as an association rather than evidence of causation. The two groups are also highly imbalanced.

## Parental Education

Mother's education (`Medu`) had a correlation of approximately:

```text
0.217
```

Father's education (`Fedu`) had a correlation of approximately:

```text
0.152
```

Both variables showed positive associations with final grade.

---

# Machine Learning Models

Three regression models were developed.

## 1. Linear Regression

Linear Regression was used as the baseline model.

## 2. Random Forest Regression

Random Forest was used because tree-based ensemble models can capture nonlinear relationships and interactions between variables.

## 3. Gradient Boosting Regression

Gradient Boosting was included as another tree-based ensemble approach for comparison.

---

# Model Evaluation

The models were evaluated using:

### Mean Absolute Error (MAE)

Measures the average absolute difference between predicted and actual grades.

**Lower is better.**

### Root Mean Squared Error (RMSE)

Measures prediction error while giving greater weight to larger errors.

**Lower is better.**

### R² Score

Measures the proportion of variation in the target variable explained by the model.

**Higher is better.**

---

## Model Comparison

| Model             |       MAE |      RMSE |        R² |
| ----------------- | --------: | --------: | --------: |
| Linear Regression |     3.395 |     4.196 |     0.141 |
| **Random Forest** | **2.966** | **3.754** | **0.313** |
| Gradient Boosting |     3.158 |     3.928 |     0.248 |

Random Forest achieved the lowest MAE and RMSE and the highest R² on the primary test split.

Therefore, Random Forest was selected as the strongest-performing model.

---

# Random Forest Hyperparameter Tuning

Grid-search hyperparameter tuning was performed on the Random Forest model.

### Best Parameters

```text
n_estimators = 300
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
```

The best cross-validation MAE obtained during tuning was:

```text
2.895
```

The tuned model produced the following test results:

```text
Test MAE:  2.986
Test RMSE: 3.765
Test R²:   0.309
```

The tuned model did not improve the original test-set performance. This result is reported rather than claiming that tuning improved the model.

---

# Cross-Validation

Random Forest was evaluated using 5-fold cross-validation.

### MAE by Fold

```text
3.066
2.758
2.720
3.282
2.675
```

### Mean Cross-Validation MAE

```text
2.900
```

### Standard Deviation

```text
0.235
```

The relatively small standard deviation indicates reasonably consistent performance across the five folds.

---

# Robustness Testing

To examine sensitivity to different train/test splits, the models were evaluated using five random states:

```text
42
10
20
30
40
```

### Random Forest Results

| Random State |   MAE |    R² |
| -----------: | ----: | ----: |
|           42 | 2.966 | 0.313 |
|           10 | 2.486 | 0.324 |
|           20 | 2.823 | 0.234 |
|           30 | 2.654 | 0.384 |
|           40 | 2.862 | 0.227 |

### Average Random Forest Performance

```text
Average MAE: 2.758
Average R²:  0.296
```

The results show that Random Forest remained the strongest model across the evaluated splits, although performance varied depending on the train/test split.

---

# Feature Importance

Two feature-importance approaches were used.

## Random Forest Feature Importance

The leading features were:

| Feature      | Importance |
| ------------ | ---------: |
| `absences`   |      0.189 |
| `failures`   |      0.145 |
| `health`     |      0.050 |
| `goout`      |      0.047 |
| `age`        |      0.038 |
| `studytime`  |      0.032 |
| `freetime`   |      0.031 |
| `traveltime` |      0.027 |
| `Walc`       |      0.027 |
| `famrel`     |      0.025 |

## Permutation Feature Importance

The leading features were:

| Feature     | Importance |
| ----------- | ---------: |
| `failures`  |      0.501 |
| `absences`  |      0.304 |
| `goout`     |      0.118 |
| `sex`       |      0.108 |
| `Mjob`      |      0.098 |
| `Medu`      |      0.084 |
| `schoolsup` |      0.077 |
| `guardian`  |      0.064 |
| `romantic`  |      0.056 |
| `studytime` |      0.050 |

Both approaches consistently identified **`failures` and `absences` as highly influential predictors**, although their exact rankings differed.

Feature importance should be interpreted as predictive contribution, not causation.

---

# Prediction Error Analysis

The model's largest prediction errors were concentrated around extreme grades.

Examples included:

```text
Actual 0  → Predicted 10.750
Actual 0  → Predicted 10.475
Actual 0  → Predicted 8.555
Actual 0  → Predicted 7.915
Actual 19 → Predicted 11.245
Actual 17 → Predicted 10.515
```

These results show that the model tended to pull extreme predictions toward the middle of the grade distribution.

This contributes to the model's overall prediction error and highlights a limitation of predicting individual students with extreme outcomes.

---

# Feature Ablation Experiment

A feature-ablation experiment was conducted by removing the two most influential predictors:

```text
failures
absences
```

### Full Model

```text
MAE:  2.966
RMSE: 3.754
R²:   0.313
```

### Reduced Model

```text
MAE:  3.494
RMSE: 4.268
R²:   0.112
```

### Change

```text
MAE increase: 0.528
R² decrease:  0.201
```

The deterioration in performance provides additional evidence that `failures` and `absences` contain substantial predictive information for final academic grade within this dataset.

The experiment does not establish causation.

---

# Key Findings

1. **Random Forest was the strongest of the three tested models.**

2. Random Forest achieved a primary test-set MAE of approximately **2.97** and R² of approximately **0.31**.

3. **Previous failures and absences were consistently among the most influential predictors.**

4. Previous failures had the strongest negative correlation with final grade among the numerical variables examined.

5. Absences had a weak linear correlation with final grade but substantial predictive importance in the Random Forest model.

6. Removing `failures` and `absences` substantially reduced model performance.

7. The model performed less accurately for students with extreme final grades.

8. Cross-validation and robustness testing showed that model performance varied across data splits but Random Forest remained the strongest overall approach.

---

# Limitations

Several limitations should be considered.

* The dataset contains only **395 students**.
* The Random Forest model's R² of approximately **0.31** indicates that substantial variation in final grades remains unexplained.
* The model struggled with some extreme grades.
* Model performance varied across different train/test splits.
* The dataset may not contain all factors that influence academic performance.
* Feature importance and correlations indicate predictive relationships or associations, not causation.
* The `higher` variable is highly imbalanced, with 375 students indicating "yes" and only 20 indicating "no."

Therefore, the model should be viewed as a demonstration of predictive modeling rather than a highly accurate system for determining an individual student's future grade.

---

# Conclusion

This project demonstrates a complete machine-learning workflow for predicting student final academic performance.

The analysis progressed from exploratory data analysis and preprocessing through model development, model comparison, hyperparameter tuning, cross-validation, robustness testing, feature-importance analysis, prediction-error analysis, and feature ablation.

Among the three evaluated models, Random Forest produced the strongest overall predictive performance. However, its moderate R² demonstrates that student academic performance is complex and cannot be fully explained using the available variables.

The project also demonstrates an important machine-learning principle: variables with weak linear correlations can still provide useful information to nonlinear models. The contrast between the weak correlation of `absences` with G3 and its strong permutation importance illustrates this point.

Overall, the project provided practical experience in data preprocessing, regression modeling, model evaluation, validation, interpretation, and critical analysis of machine-learning results.

---

# Project Structure

```text
student-performance-ml/
│
├── student_analysis.py
├── student-mat.csv
├── README.md
│
└── graphs/
    ├── absences_vs_final_grade.png
    ├── actual_vs_predicted_grades.png
    ├── failures_vs_final.png
    ├── father_education_vs_final_grade.png
    ├── final_grade_distribution.png
    ├── higher_education_vs_final_grade.png
    ├── model_performance_comparison.png
    ├── model_performance_r2.png
    ├── mother_education_vs_final_grade.png
    ├── random_forest_feature_importance.png
    └── studytime_vs_final_grade.png
```

---

# How to Run

### 1. Clone or download the project

Place the project files in the same project directory.

### 2. Install the required Python libraries

The project uses common data-science and machine-learning libraries, including:

```text
pandas
numpy
matplotlib
scikit-learn
```

### 3. Run the analysis

From the project directory:

```bash
python student_analysis.py
```

The program performs the analysis and generates the project's visualizations inside the `graphs` folder.

---

# Skills Demonstrated

This project demonstrates practical experience with:

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Exploratory Data Analysis
* Data preprocessing
* Feature engineering
* Regression
* Random Forest
* Gradient Boosting
* Linear Regression
* Hyperparameter tuning
* GridSearchCV
* Cross-validation
* Model evaluation
* Feature importance
* Permutation importance
* Error analysis
* Feature ablation
* Reproducible machine-learning workflows
* Data visualization

---

# Final Model

**Selected model: Random Forest Regression**

Primary test performance:

```text
MAE:  2.966
RMSE: 3.754
R²:   0.313
```

The model provides useful predictive information but should not be interpreted as a highly accurate predictor of individual student outcomes.
