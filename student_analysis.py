import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# Load the Mathematics dataset
df = pd.read_csv("student-mat.csv", sep=";")
print("\n--- DATASET SIZE CHECK ---")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))

print("Dataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nFinal grade statistics:")
print(df["G3"].describe())

print("\nNumber of students for each final grade:")
print(df["G3"].value_counts().sort_index())
print("\n--- FINAL GRADE ANALYSIS ---")

print("\nNumber of students for each final grade:")
print(df["G3"].value_counts().sort_index())

print("\nFinal grade statistics:")
print(df["G3"].describe())

print("\nAverage final grade:")
print(df["G3"].mean())

print("\nMedian final grade:")
print(df["G3"].median())


print("\nCreating final grade graph...")

plt.figure(figsize=(10, 6))

df["G3"].value_counts().sort_index().plot(kind="bar")

plt.title("Distribution of Final Mathematics Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.xticks(rotation=0)

plt.tight_layout()

# Save the graph as an image
plt.savefig("graphs/final_grade_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

print("Graph saved as final_grade_distribution.png")
print("\n--- STUDY TIME VS FINAL GRADE ---")

studytime_analysis = df.groupby("studytime")["G3"].agg(
    ["count", "mean", "median", "min", "max"]
)

print(studytime_analysis)


studytime_means = df.groupby("studytime")["G3"].mean()

plt.figure(figsize=(10, 6))

studytime_means.plot(kind="bar")

plt.title("Average Final Grade by Weekly Study Time")
plt.xlabel("Study Time Category")
plt.ylabel("Average Final Grade (G3)")
plt.xticks(
    ticks=range(4),
    labels=[
        "Less than 2 hours",
        "2–5 hours",
        "5–10 hours",
        "More than 10 hours"
    ],
    rotation=0
)

plt.tight_layout()

plt.savefig("graphs/studytime_vs_final_grade.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nGraph saved as studytime_vs_final_grade.png")
print("\n--- PREVIOUS FAILURES VS FINAL GRADE ---")

failures_analysis = df.groupby("failures")["G3"].agg(
    ["count", "mean", "median", "min", "max"]
)

print(failures_analysis)
failures_means = df.groupby("failures")["G3"].mean()

plt.figure(figsize=(10, 6))

failures_means.plot(kind="bar")

plt.title("Average Final Grade by Number of Previous Failures")
plt.xlabel("Number of Previous Failures")
plt.ylabel("Average Final Grade (G3)")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig("graphs/failures_vs_final.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nGraph saved as failures_vs_final.png")
print("\n--- ABSENCES VS FINAL GRADE ---")

absence_correlation = df["absences"].corr(df["G3"])

print("Correlation between absences and final grade:")
print(absence_correlation)

# Create a scatter plot
plt.figure(figsize=(10, 6))

plt.scatter(df["absences"], df["G3"])

plt.title("Absences vs Final Mathematics Grade")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")

plt.tight_layout()

plt.savefig(
    "graphs/absences_vs_final_grade.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\nGraph saved as absences_vs_final_grade.png")

# Create a scatter plot

plt.figure(figsize=(10, 6))

plt.scatter(df["absences"], df["G3"])

plt.title("Absences vs Final Mathematics Grade")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")

plt.tight_layout()

# Save the graph
plt.savefig("graphs/absences_vs_final_grade.png", dpi=300, bbox_inches="tight")
plt.close()

print("\n--- MOTHER'S EDUCATION VS FINAL GRADE ---")

medu_analysis = df.groupby("Medu")["G3"].agg(
    ["count", "mean", "median", "min", "max"]
)

print(medu_analysis)
medu_means = df.groupby("Medu")["G3"].mean()

plt.figure(figsize=(10, 6))

medu_means.plot(kind="bar")

plt.title("Average Final Grade by Mother's Education")
plt.xlabel("Mother's Education Level")
plt.ylabel("Average Final Grade (G3)")
plt.xticks(
    ticks=range(5),
    labels=[
        "None",
        "Primary",
        "5th–9th Grade",
        "Secondary",
        "Higher Education"
    ],
    rotation=0
)

plt.tight_layout()

plt.savefig("graphs/mother_education_vs_final_grade.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nGraph saved as mother_education_vs_final_grade.png")
print("\n--- FATHER'S EDUCATION VS FINAL GRADE ---")

fedu_analysis = df.groupby("Fedu")["G3"].agg(
    ["count", "mean", "median", "min", "max"]
)

print(fedu_analysis)
fedu_means = df.groupby("Fedu")["G3"].mean()

plt.figure(figsize=(10, 6))

fedu_means.plot(kind="bar")

plt.title("Average Final Grade by Father's Education")
plt.xlabel("Father's Education Level")
plt.ylabel("Average Final Grade (G3)")
plt.xticks(
    ticks=range(5),
    labels=[
        "None",
        "Primary",
        "5th–9th Grade",
        "Secondary",
        "Higher Education"
    ],
    rotation=0
)

plt.tight_layout()

plt.savefig("graphs/father_education_vs_final_grade.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nGraph saved as father_education_vs_final_grade.png")
print("\n--- CORRELATION WITH FINAL GRADE ---")

numeric_features = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences"
]

correlations = df[numeric_features + ["G3"]].corr()["G3"].drop("G3")

print("\nCorrelation of each numerical variable with G3:")
print(correlations.sort_values(ascending=False))
print("\n--- HIGHER EDUCATION ASPIRATION VS FINAL GRADE ---")

higher_analysis = df.groupby("higher")["G3"].agg(
    ["count", "mean", "median", "min", "max"]
)

print(higher_analysis)
higher_means = df.groupby("higher")["G3"].mean()

plt.figure(figsize=(8, 6))

higher_means.plot(kind="bar")

plt.title("Average Final Grade by Higher Education Aspiration")
plt.xlabel("Wants Higher Education")
plt.ylabel("Average Final Grade (G3)")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig("graphs/higher_education_vs_final_grade.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nGraph saved as higher_education_vs_final_grade.png")
print("\n--- MACHINE LEARNING DATA PREPARATION ---")

# Remove G1 and G2 because they are previous-period grades
# and should not be used as predictors in our research question.

X = df.drop(columns=["G1", "G2", "G3"])

# G3 is our target variable
y = df["G3"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

print("\nNumber of features:")
print(X.shape[1])

print("\nNumber of target values:")
print(y.shape[0])
# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=["str"]).columns.tolist()
numerical_columns = X.select_dtypes(exclude=["object"]).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)
from sklearn.preprocessing import StandardScaler

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ("numerical", "passthrough", numerical_columns)
    ]
)

print("\nPreprocessing pipeline created successfully.")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n--- TRAIN/TEST SPLIT ---")

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

# Create the machine learning pipeline
linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

# Train the model
linear_model.fit(X_train, y_train)

print("\n--- LINEAR REGRESSION MODEL ---")
print("Model trained successfully!")
# Make predictions on the test data
y_pred = linear_model.predict(X_test)

print("\n--- MODEL PREDICTIONS ---")

print("Actual grades:")
print(y_test.head(10).values)

print("\nPredicted grades:")
print(y_pred[:10])

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print("\n--- MODEL EVALUATION ---")

print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)

# Create the Random Forest pipeline
random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ))
    ]
)

# Train the Random Forest model
random_forest_model.fit(X_train, y_train)

print("\n--- RANDOM FOREST REGRESSION MODEL ---")
print("Model trained successfully!")

# Make predictions
rf_pred = random_forest_model.predict(X_test)

# Evaluate the Random Forest
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\n--- RANDOM FOREST EVALUATION ---")

print("Mean Absolute Error (MAE):", rf_mae)
print("Root Mean Squared Error (RMSE):", rf_rmse)
print("R² Score:", rf_r2)
# Get the trained Random Forest model
rf_model = random_forest_model.named_steps["model"]

# Get the preprocessor
rf_preprocessor = random_forest_model.named_steps["preprocessor"]

# Get feature names after one-hot encoding
feature_names = rf_preprocessor.get_feature_names_out()

# Get feature importance values
feature_importances = rf_model.feature_importances_

# Create a DataFrame containing feature names and importance
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": feature_importances
})

# Sort from most important to least important
importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\n--- TOP 15 FEATURE IMPORTANCES ---")
print(importance_df.head(15))
# Plot the top 15 features
top_features = importance_df.head(15).sort_values(
    by="importance"
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.title("Top 15 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig("graphs/random_forest_feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nGraph saved as random_forest_feature_importance.png")
print("\n--- RANDOM FOREST PREDICTION SUMMARY ---")
# Compare actual and predicted grades
comparison = pd.DataFrame({
    "Actual G3": y_test.values,
    "Predicted G3": rf_pred
})

print("\n--- ACTUAL VS PREDICTED GRADES ---")
print(comparison.head(15))

# Calculate prediction errors
comparison["Error"] = (
    comparison["Actual G3"] - comparison["Predicted G3"]
)

comparison["Absolute Error"] = abs(comparison["Error"])

print("\n--- PREDICTION ERROR SUMMARY ---")
print(comparison.head(15))

print("\nAverage absolute prediction error:")
print(comparison["Absolute Error"].mean())
from sklearn.model_selection import GridSearchCV

print("\n--- RANDOM FOREST HYPERPARAMETER TUNING ---")

# Create a Random Forest pipeline for tuning
rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42))
    ]
)

# Parameters we want to test
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 5, 10, 15],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}

# Grid search using only the training data
grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=1
)

# Train and search for the best combination
grid_search.fit(X_train, y_train)

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation MAE:")
print(-grid_search.best_score_)
# Get the best tuned Random Forest model
best_rf_model = grid_search.best_estimator_

# Make predictions on the untouched test set
tuned_rf_pred = best_rf_model.predict(X_test)

# Evaluate the tuned model
tuned_mae = mean_absolute_error(y_test, tuned_rf_pred)
tuned_rmse = np.sqrt(mean_squared_error(y_test, tuned_rf_pred))
tuned_r2 = r2_score(y_test, tuned_rf_pred)

print("\n--- TUNED RANDOM FOREST TEST RESULTS ---")

print("Test MAE:", tuned_mae)
print("Test RMSE:", tuned_rmse)
print("Test R²:", tuned_r2)
from sklearn.model_selection import cross_val_score

print("\n--- RANDOM FOREST 5-FOLD CROSS-VALIDATION ---")

cv_scores = cross_val_score(
    random_forest_model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=1
)

cv_mae_scores = -cv_scores

print("MAE for each fold:")
print(cv_mae_scores)

print("\nMean Cross-Validation MAE:")
print(cv_mae_scores.mean())

print("\nStandard Deviation of MAE:")
print(cv_mae_scores.std())
# Create absence groups
df["absence_group"] = pd.cut(
    df["absences"],
    bins=[-1, 5, 10, 20, 30, float("inf")],
    labels=["0-5", "6-10", "11-20", "21-30", "31+"]
)

print("\n--- ABSENCE GROUP VS FINAL GRADE ---")

absence_analysis = df.groupby(
    "absence_group",
    observed=True
)["G3"].agg(["count", "mean", "median", "min", "max"])

print(absence_analysis)
from sklearn.ensemble import GradientBoostingRegressor

print("\n--- GRADIENT BOOSTING REGRESSION MODEL ---")

# Create Gradient Boosting pipeline
gradient_boosting_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ))
    ]
)

# Train the model
gradient_boosting_model.fit(X_train, y_train)

print("Model trained successfully!")

# Make predictions
gb_pred = gradient_boosting_model.predict(X_test)

# Evaluate the model
gb_mae = mean_absolute_error(y_test, gb_pred)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
gb_r2 = r2_score(y_test, gb_pred)

print("\n--- GRADIENT BOOSTING EVALUATION ---")
print("Mean Absolute Error (MAE):", gb_mae)
print("Root Mean Squared Error (RMSE):", gb_rmse)
print("R² Score:", gb_r2)
# ============================================================
# ROBUSTNESS TESTING
# ============================================================

print("\n--- ROBUSTNESS TESTING ---")

random_states = [42, 10, 20, 30, 40]

results = []

for seed in random_states:

    # Create a new train/test split
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed
    )

    # -----------------------------
    # Linear Regression
    # -----------------------------
    lr_model_r = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ]
    )

    lr_model_r.fit(X_train_r, y_train_r)
    lr_pred_r = lr_model_r.predict(X_test_r)

    lr_mae_r = mean_absolute_error(y_test_r, lr_pred_r)
    lr_rmse_r = np.sqrt(mean_squared_error(y_test_r, lr_pred_r))
    lr_r2_r = r2_score(y_test_r, lr_pred_r)

    # -----------------------------
    # Random Forest
    # -----------------------------
    rf_model_r = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=200,
                random_state=42
            ))
        ]
    )

    rf_model_r.fit(X_train_r, y_train_r)
    rf_pred_r = rf_model_r.predict(X_test_r)

    rf_mae_r = mean_absolute_error(y_test_r, rf_pred_r)
    rf_rmse_r = np.sqrt(mean_squared_error(y_test_r, rf_pred_r))
    rf_r2_r = r2_score(y_test_r, rf_pred_r)

    # -----------------------------
    # Gradient Boosting
    # -----------------------------
    gb_model_r = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            ))
        ]
    )

    gb_model_r.fit(X_train_r, y_train_r)
    gb_pred_r = gb_model_r.predict(X_test_r)

    gb_mae_r = mean_absolute_error(y_test_r, gb_pred_r)
    gb_rmse_r = np.sqrt(mean_squared_error(y_test_r, gb_pred_r))
    gb_r2_r = r2_score(y_test_r, gb_pred_r)

    # Store results
    results.append({
        "Random State": seed,

        "Linear Regression MAE": lr_mae_r,
        "Linear Regression R2": lr_r2_r,

        "Random Forest MAE": rf_mae_r,
        "Random Forest R2": rf_r2_r,

        "Gradient Boosting MAE": gb_mae_r,
        "Gradient Boosting R2": gb_r2_r
    })


# Convert results into a DataFrame
robustness_df = pd.DataFrame(results)

print("\n--- ROBUSTNESS RESULTS ---")
print(robustness_df.to_string(index=False))


# Calculate average performance
print("\n--- AVERAGE PERFORMANCE ACROSS SPLITS ---")

print(
    "\nLinear Regression Average MAE:",
    robustness_df["Linear Regression MAE"].mean()
)

print(
    "Random Forest Average MAE:",
    robustness_df["Random Forest MAE"].mean()
)

print(
    "Gradient Boosting Average MAE:",
    robustness_df["Gradient Boosting MAE"].mean()
)

print(
    "\nLinear Regression Average R²:",
    robustness_df["Linear Regression R2"].mean()
)

print(
    "Random Forest Average R²:",
    robustness_df["Random Forest R2"].mean()
)

print(
    "Gradient Boosting Average R²:",
    robustness_df["Gradient Boosting R2"].mean()
)

print("\n--- PERMUTATION FEATURE IMPORTANCE ---")

# Calculate permutation importance on the test data
perm_result = permutation_importance(
    random_forest_model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="neg_mean_absolute_error",
    n_jobs=1
)

# Permutation importance works with the original input features
feature_names = X_test.columns

# Create a DataFrame
permutation_df = pd.DataFrame({
    "feature": feature_names,
    "importance": perm_result.importances_mean
})

# Sort from most important to least important
permutation_df = permutation_df.sort_values(
    by="importance",
    ascending=False
)

print("\nTOP 15 PERMUTATION FEATURE IMPORTANCES:")
print(permutation_df.head(15).to_string(index=False))

print("\n--- ACTUAL VS PREDICTED GRADES ---")

# Use the original Random Forest model
final_predictions = random_forest_model.predict(X_test)

# Create scatter plot
plt.figure(figsize=(8, 6))

plt.scatter(y_test, final_predictions, alpha=0.7)

# Perfect prediction reference line
min_grade = min(y_test.min(), final_predictions.min())
max_grade = max(y_test.max(), final_predictions.max())

plt.plot(
    [min_grade, max_grade],
    [min_grade, max_grade],
    linestyle="--"
)

plt.xlabel("Actual Final Grade (G3)")
plt.ylabel("Predicted Final Grade")
plt.title("Actual vs Predicted Final Grades")

plt.tight_layout()

plt.savefig("graphs/actual_vs_predicted_grades.png", dpi=300)
plt.close()

print("Graph saved as actual_vs_predicted_grades.png")
print("\n--- LARGEST PREDICTION ERRORS ---")

# Create a DataFrame containing actual and predicted grades
error_analysis = pd.DataFrame({
    "Actual G3": y_test.values,
    "Predicted G3": np.round(final_predictions, 3)
})

# Calculate prediction error
error_analysis["Error"] = (
    error_analysis["Actual G3"] -
    error_analysis["Predicted G3"]
)

# Calculate absolute error
error_analysis["Absolute Error"] = (
    error_analysis["Error"].abs()
)

# Sort by largest prediction error
largest_errors = error_analysis.sort_values(
    by="Absolute Error",
    ascending=False
)

print("\nTOP 10 LARGEST PREDICTION ERRORS:")
print(largest_errors.head(10).to_string(index=False))
# ============================================================
# PHASE 11: FEATURE ABLATION EXPERIMENT
# ============================================================

print("\n--- FEATURE ABLATION EXPERIMENT ---")

# Full feature set
X_full = X.copy()

# Remove the two strongest predictors
X_reduced = X.drop(columns=["failures", "absences"])

# Use the same train/test split for a fair comparison
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full,
    y,
    test_size=0.20,
    random_state=42
)

X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(
    X_reduced,
    y,
    test_size=0.20,
    random_state=42
)

# ------------------------------------------------------------
# FULL RANDOM FOREST MODEL
# ------------------------------------------------------------

full_rf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ))
    ]
)

full_rf.fit(X_train_full, y_train_full)

full_predictions = full_rf.predict(X_test_full)

full_mae = mean_absolute_error(y_test_full, full_predictions)
full_rmse = np.sqrt(mean_squared_error(y_test_full, full_predictions))
full_r2 = r2_score(y_test_full, full_predictions)

print("\nFULL MODEL:")
print("MAE:", full_mae)
print("RMSE:", full_rmse)
print("R²:", full_r2)


# ------------------------------------------------------------
# REDUCED RANDOM FOREST MODEL
# ------------------------------------------------------------

# Create a new preprocessor for the reduced feature set
categorical_reduced = X_reduced.select_dtypes(
    include=["object", "string"]
).columns

numerical_reduced = X_reduced.select_dtypes(
    exclude=["object", "string"]
).columns

preprocessor_reduced = ColumnTransformer(
    transformers=[
        ("numerical",
         "passthrough",
         numerical_reduced),

        ("categorical",
         OneHotEncoder(handle_unknown="ignore"),
         categorical_reduced)
    ]
)

reduced_rf = Pipeline(
    steps=[
        ("preprocessor", preprocessor_reduced),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ))
    ]
)

reduced_rf.fit(X_train_reduced, y_train_reduced)

reduced_predictions = reduced_rf.predict(X_test_reduced)

reduced_mae = mean_absolute_error(
    y_test_reduced,
    reduced_predictions
)

reduced_rmse = np.sqrt(
    mean_squared_error(
        y_test_reduced,
        reduced_predictions
    )
)

reduced_r2 = r2_score(
    y_test_reduced,
    reduced_predictions
)

print("\nREDUCED MODEL (WITHOUT FAILURES AND ABSENCES):")
print("MAE:", reduced_mae)
print("RMSE:", reduced_rmse)
print("R²:", reduced_r2)


# ------------------------------------------------------------
# COMPARISON
# ------------------------------------------------------------

print("\n--- ABLATION COMPARISON ---")

print(
    "MAE change:",
    reduced_mae - full_mae
)

print(
    "R² change:",
    reduced_r2 - full_r2
)
# ============================================================
# MODEL PERFORMANCE COMPARISON
# ============================================================

print("\n--- MODEL PERFORMANCE COMPARISON ---")

model_names = [
    "Linear Regression",
    "Random Forest",
    "Gradient Boosting"
]

model_mae = [
    mae,
    rf_mae,
    gb_mae
]

plt.figure(figsize=(10, 6))

bars = plt.bar(model_names, model_mae)

plt.title("Model Performance Comparison - MAE")
plt.xlabel("Machine Learning Model")
plt.ylabel("Mean Absolute Error (MAE)")
plt.xticks(rotation=15)

# Display values above each bar
for bar, value in zip(bars, model_mae):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    "graphs/model_performance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Graph saved as model_performance_comparison.png")
# ============================================================
# MODEL PERFORMANCE COMPARISON - R²
# ============================================================

print("\n--- MODEL PERFORMANCE COMPARISON - R² ---")

model_names = [
    "Linear Regression",
    "Random Forest",
    "Gradient Boosting"
]

model_r2 = [
    r2,
    rf_r2,
    gb_r2
]

plt.figure(figsize=(10, 6))

bars = plt.bar(model_names, model_r2)

plt.title("Model Performance Comparison - R²")
plt.xlabel("Machine Learning Model")
plt.ylabel("R² Score")
plt.xticks(rotation=15)

# Display values above each bar
for bar, value in zip(bars, model_r2):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    "graphs/model_performance_r2.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Graph saved as model_performance_r2.png")