# Student Performance Prediction Using Machine Learning

A machine learning project that analyzes and predicts students' final mathematics grades using demographic, family, social, and academic-related factors.

The project compares multiple regression models and investigates which features contribute most to prediction performance.

---

## 📌 Project Overview

This project applies a complete machine learning workflow to the **Student Performance Dataset**.

The analysis includes:

* Data exploration and preprocessing
* Exploratory Data Analysis (EDA)
* Regression modeling
* Model comparison
* Hyperparameter tuning
* Cross-validation
* Robustness testing
* Feature importance analysis
* Feature ablation
* Prediction error analysis

The goal is to understand the factors associated with student performance while building a model capable of predicting final mathematics grades.

---

## 🎯 Project Objectives

The main objectives were to:

* Explore patterns in students' final mathematics grades.
* Identify factors associated with academic performance.
* Build machine learning models for predicting final grades.
* Compare Linear Regression, Random Forest, and Gradient Boosting.
* Evaluate models using MAE, RMSE, and R².
* Tune the Random Forest model using cross-validation.
* Test model robustness across different train/test splits.
* Identify important predictive features.
* Evaluate the contribution of key features through feature ablation.

---

## 📊 Dataset

The project uses the **Student Performance Dataset**, containing demographic, family, social, and academic information about students.

### Dataset Summary

| Property            | Value |
| ------------------- | ----: |
| Students            |   395 |
| Original columns    |    33 |
| Input features used |    30 |
| Target variable     |    G3 |
| Target range        |  0–20 |

The target variable, `G3`, represents the student's final mathematics grade.

### Important Modeling Decision: Preventing Target Leakage

The variables `G1` and `G2` were intentionally removed from the machine-learning features.

Although these variables are strong predictors of `G3`, including them would introduce **target leakage** because they represent earlier grades from the same course.

Therefore:

```python
X = df.drop(columns=["G1", "G2", "G3"])
y = df["G3"]
```

This allows the model to focus on demographic, behavioral, family, and other available student characteristics.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Git
* GitHub

---

## 🔍 Exploratory Data Analysis

Several analyses were performed to understand the dataset and investigate relationships with final grades.

### Final Grade Distribution

The final mathematics grade ranged from **0 to 20**.

| Statistic           | Value |
| ------------------- | ----: |
| Mean                | 10.42 |
| Median              |    11 |
| Most frequent grade |    10 |

![Final Grade Distribution](graphs/final_grade_distribution.png)

### Previous Failures

Previous failures showed one of the strongest relationships with final performance.

| Previous Failures | Average Final Grade |
| ----------------: | ------------------: |
|                 0 |               11.25 |
|                 1 |                8.12 |
|                 2 |                6.24 |
|                 3 |                5.69 |

Students with more previous failures generally had lower final grades.

![Failures vs Final Grade](graphs/failures_vs_final.png)

### Study Time

Students reporting higher study-time categories generally had somewhat higher average final grades.

| Study Time | Average Final Grade |
| ---------: | ------------------: |
|          1 |               10.05 |
|          2 |               10.17 |
|          3 |               11.40 |
|          4 |               11.26 |

![Study Time vs Final Grade](graphs/studytime_vs_final_grade.png)

### Higher Education Aspiration

Students who indicated that they wanted to pursue higher education had a higher average final grade.

| Higher Education Aspiration | Average Final Grade |
| --------------------------- | ------------------: |
| No                          |                6.80 |
| Yes                         |               10.61 |

![Higher Education vs Final Grade](graphs/higher_education_vs_final_grade.png)

### Absences

The simple correlation between absences and final grade was approximately:

```text
0.034
```

This indicates a very weak linear relationship when considering the entire dataset.

However, the machine-learning experiments showed that `absences` still provided useful predictive information when combined with other features.

![Absences vs Final Grade](graphs/absences_vs_final_grade.png)

---

## 🤖 Machine Learning Models

Three regression algorithms were trained and evaluated.

### 1. Linear Regression

Used as a baseline model to establish a simple linear benchmark.

### 2. Random Forest Regression

Used to capture nonlinear relationships and interactions between student characteristics.

### 3. Gradient Boosting Regression

Used as another tree-based ensemble approach for comparison.

### Train/Test Split

The dataset was divided into:

* **80% training data:** 316 students
* **20% testing data:** 79 students

A `random_state` of **42** was used for the main train/test split.

---

## 📈 Model Performance

The models were evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

### Test Set Results

| Model             |       MAE |      RMSE |        R² |
| ----------------- | --------: | --------: | --------: |
| Linear Regression |     3.395 |     4.196 |     0.141 |
| **Random Forest** | **2.966** | **3.754** | **0.313** |
| Gradient Boosting |     3.158 |     3.928 |     0.248 |

![Model Performance Comparison](graphs/model_performance_comparison.png)

### Best Performing Model

The **Random Forest Regressor** achieved the strongest performance on the test set.

| Metric | Random Forest |
| ------ | ------------: |
| MAE    |         2.966 |
| RMSE   |         3.754 |
| R²     |         0.313 |

The Random Forest model generally predicted final grades more accurately than the Linear Regression and Gradient Boosting models tested in this project.

---

## ⚙️ Hyperparameter Tuning

Grid Search with 5-fold cross-validation was used to tune the Random Forest model.

### Best Parameters

```text
n_estimators = 300
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
```

The best cross-validation MAE was approximately:

```text
2.895
```

The tuned Random Forest achieved:

```text
MAE:  2.986
RMSE: 3.765
R²:   0.309
```

The tuned model performed very similarly to the original Random Forest, suggesting that the initial model was already reasonably well configured for this dataset.

---

## 🔄 Cross-Validation

A 5-fold cross-validation experiment was performed on the Random Forest model.

| Metric             | Value |
| ------------------ | ----: |
| Mean CV MAE        | 2.900 |
| Standard Deviation | 0.235 |

The relatively small standard deviation indicates that model performance was reasonably consistent across the five folds.

---

## 🧪 Robustness Testing

To determine whether model performance depended heavily on a single train/test split, the models were evaluated using multiple random states:

```text
42, 10, 20, 30, 40
```

### Average Performance Across Splits

| Model             | Average MAE | Average R² |
| ----------------- | ----------: | ---------: |
| Linear Regression |       3.272 |      0.037 |
| **Random Forest** |   **2.758** |  **0.296** |
| Gradient Boosting |       2.917 |      0.230 |

Random Forest maintained the best average MAE and average R² across the tested splits.

---

## 🔎 Feature Importance

Two approaches were used to investigate feature importance:

1. Random Forest built-in feature importance
2. Permutation feature importance

### Permutation Feature Importance

The most influential original features included:

| Feature   | Permutation Importance |
| --------- | ---------------------: |
| failures  |                  0.501 |
| absences  |                  0.304 |
| goout     |                  0.118 |
| sex       |                  0.108 |
| Mjob      |                  0.098 |
| Medu      |                  0.084 |
| schoolsup |                  0.077 |

`failures` and `absences` provided particularly useful predictive information in the model.

![Random Forest Feature Importance](graphs/random_forest_feature_importance.png)

---

## 🧩 Feature Ablation Experiment

A feature ablation experiment was performed to measure the contribution of `failures` and `absences`.

The full model was compared with a reduced model that excluded both features.

### Results

| Model                           |       MAE |      RMSE |        R² |
| ------------------------------- | --------: | --------: | --------: |
| Full Model                      | **2.966** | **3.754** | **0.313** |
| Without `failures` & `absences` |     3.492 |     4.264 |     0.113 |

Removing the two features resulted in:

| Change       | Value |
| ------------ | ----: |
| MAE increase | 0.526 |
| R² decrease  | 0.199 |

This provides additional evidence that `failures` and `absences` contribute substantially to the model's predictive performance.

---

## 🎯 Prediction Error Analysis

The project also examined the largest differences between actual and predicted grades.

The model struggled particularly with some extreme outcomes, including students with very low actual grades and some students with very high grades.

For example, some students with an actual grade of **0** were predicted to have grades around **8–11**.

This highlights an important limitation: the model performs better around the central range of the grade distribution and has difficulty predicting some extreme cases.

![Actual vs Predicted Grades](graphs/actual_vs_predicted_grades.png)

---

## 📊 Visualizations

The project generates several visualizations, including:

* Final grade distribution
* Study time vs. final grade
* Previous failures vs. final grade
* Absences vs. final grade
* Mother's education vs. final grade
* Father's education vs. final grade
* Higher education aspiration vs. final grade
* Random Forest feature importance
* Actual vs. predicted grades
* Model performance comparison
* Model R² comparison

All generated graphs are stored in the `graphs/` directory.

---

## 📁 Project Structure

```text
student-performance-ml/
│
├── graphs/
│   ├── absences_vs_final_grade.png
│   ├── actual_vs_predicted_grades.png
│   ├── failures_vs_final.png
│   ├── father_education_vs_final_grade.png
│   ├── final_grade_distribution.png
│   ├── higher_education_vs_final_grade.png
│   ├── model_performance_comparison.png
│   ├── model_performance_r2.png
│   ├── mother_education_vs_final_grade.png
│   ├── random_forest_feature_importance.png
│   └── studytime_vs_final_grade.png
│
├── student-mat.csv
├── student_analysis.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/eutor10/student-performance-ml.git
cd student-performance-ml
```

### 2. Install the required packages

```bash
pip install -r requirements.txt
```

### 3. Run the analysis

```bash
python student_analysis.py
```

The script performs the complete analysis and generates visualizations inside the `graphs/` directory.

---

## 💡 Key Findings

1. **Random Forest performed best** among the three tested models.
2. Previous failures showed a strong negative relationship with final academic performance.
3. `failures` and `absences` were among the most important features in the Random Forest analysis.
4. Removing `failures` and `absences` substantially reduced model performance.
5. Random Forest maintained the strongest average performance across multiple train/test splits.
6. The model has difficulty predicting some extreme student outcomes.
7. Earlier grades (`G1` and `G2`) were excluded to prevent target leakage.

---

## ⚠️ Limitations

This project has several limitations:

* The dataset contains only 395 students.
* The model explains only part of the variation in final grades.
* Some important factors affecting academic performance may not be included.
* The dataset represents students from a specific educational context and may not generalize to every population.
* Predicting extreme grades remains challenging.

Therefore, the model should be viewed as an **analytical and educational machine-learning project**, rather than a system for making high-stakes decisions about students.

---

## 🚀 Future Improvements

Possible future improvements include:

* Testing additional regression algorithms.
* Applying more systematic feature engineering.
* Exploring regularization techniques.
* Performing more extensive hyperparameter optimization.
* Using learning curves to investigate whether additional data could improve performance.
* Evaluating the model on an independent external dataset.
* Building an interactive dashboard for predictions and data exploration.
* Deploying the model as a simple web application.

---

## 📚 Skills Demonstrated

This project demonstrates practical experience with:

* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Data visualization
* Feature selection
* Prevention of target leakage
* Regression modeling
* Ensemble machine learning
* Model evaluation
* Hyperparameter tuning
* Cross-validation
* Robustness testing
* Feature importance
* Permutation importance
* Feature ablation
* Error analysis
* Python
* Scikit-learn
* Git and GitHub

---

## 👤 Author

**Eutor S. Momolu**

This project was developed as part of my practical machine-learning and data analytics portfolio.
