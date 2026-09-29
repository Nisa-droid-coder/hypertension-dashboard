# 🫀 Hypertension Risk Predictor Dashboard

An interactive machine learning dashboard for exploring hypertension risk factors, comparing predictive models, interpreting model behavior, and estimating hypertension risk from lifestyle and demographic information.

Built with Python and Streamlit. (https://hypertension-dashboard.streamlit.app/)

---

## 📌 Project Overview

Hypertension is influenced by multiple demographic and lifestyle factors, including age, body mass index (BMI), salt intake, physical activity, smoking, stress, sleep, and family history.

The **Hypertension Risk Predictor Dashboard** was developed to provide an interactive environment where users can:

- Upload a hypertension dataset
- Explore patient characteristics
- Analyze potential hypertension risk factors
- Compare hypertension prevalence across different age groups
- Train and evaluate multiple machine learning models
- Interpret predictions using SHAP
- Examine model learning curves and confusion matrices
- Estimate an individual's hypertension risk based on lifestyle information

The project focuses particularly on **lifestyle-based prediction**.

Columns such as `BP_History` and `Medication`, when present, are intentionally excluded from the predictive features to reduce the risk of target leakage and allow the models to focus on demographic and lifestyle predictors.

> ⚠️ **Important:** This project is intended for educational, research, and data-analysis purposes only. It is not a medical diagnostic system and should not be used as a replacement for professional medical advice.

---

# 🎯 Project Objectives

The main objectives of this project are:

1. Explore relationships between lifestyle factors and hypertension.
2. Visualize hypertension patterns across different populations and age groups.
3. Train predictive machine learning models using selected lifestyle and demographic variables.
4. Compare model performance using multiple evaluation metrics.
5. Improve model transparency using SHAP explainability.
6. Provide an interactive risk assessment interface.
7. Demonstrate a complete machine learning workflow through a Streamlit web application.

---

# ✨ Key Features

## 📁 1. Dataset Upload and Validation

Users can upload their own hypertension dataset in CSV format.

Before analysis begins, the dashboard checks that the required columns are available and performs basic validation on numerical fields.

The required variables are:

| Variable | Description |
|---|---|
| `Age` | Age of the individual |
| `Salt_Intake` | Estimated daily salt intake |
| `Stress_Score` | Stress score from 0 to 10 |
| `Sleep_Duration` | Average sleep duration |
| `BMI` | Body Mass Index |
| `Family_History` | Family history of hypertension |
| `Exercise_Level` | Physical activity level |
| `Smoking_Status` | Smoking status |
| `Has_Hypertension` | Target variable indicating hypertension |

The application also recognizes:

- `BP_History`
- `Medication`

These variables are not included among the model's predictive lifestyle features.

---

# 📊 2. Dataset Overview

After uploading a valid dataset, users can examine:

- Total number of records
- Number of records after filtering
- Hypertension prevalence
- Average patient age
- Data types
- Unique values
- Missing values
- Descriptive statistics
- Sample records

This page gives users a quick understanding of the structure and quality of the uploaded dataset.

---

# 🔍 3. Exploratory Data Analysis

The dashboard provides interactive visualizations to investigate the relationship between hypertension and several risk factors.

Analysis includes:

### Salt Intake
Examines salt intake distributions across age groups and hypertension status.

### Body Mass Index
Compares BMI categories including:

- Underweight
- Normal
- Overweight
- Obese

### Stress Score
Examines differences between low, moderate, high, and very high stress levels.

### Sleep Duration
Compares hypertension outcomes across different sleep-duration categories.

### Hypertension by Age
Displays hypertension prevalence across dynamically generated age groups.

### Categorical Variables

The application also evaluates:

- Family history
- Exercise level
- Smoking status

Cross-tabulations allow users to compare hypertension prevalence for each category.

---

# 👥 Age-Based Analysis

The application contains several age-based analytical features.

Users can select an age range directly from the sidebar.

The selected range affects both:

- Data exploration
- Machine learning model training

Models automatically retrain when the selected age range changes.

This makes it possible to investigate whether the importance and predictive behavior of lifestyle factors differ between age groups.

---

# 📈 Advanced Age Analysis

The dashboard provides an additional age analysis containing:

### Age Distribution

A histogram displays the age distribution of the selected population.

A Kernel Density Estimate may also be displayed when appropriate.

### Hypertension Trend by Age

Hypertension prevalence is calculated across age decades such as:

- <20
- 20-29
- 30-39
- 40-49
- 50-59
- 60-69
- 70-79
- 80+

The application also attempts to estimate where the observed hypertension rate crosses the **50% prevalence level** using interpolation between age groups.

> This value describes the pattern observed in the uploaded dataset. It should not be interpreted as a clinical age threshold for an individual.

---

# 🤖 Machine Learning Models

The dashboard currently supports three classification algorithms.

## 1. Logistic Regression

Logistic Regression provides a relatively interpretable baseline classification model.

Before training Logistic Regression, numerical and encoded features are standardized using:

```python
StandardScaler()
```

Standardization helps ensure that features measured on different scales contribute appropriately to the Logistic Regression model.

---

## 2. Random Forest

Random Forest is an ensemble-based classification algorithm that combines multiple decision trees to make predictions.

In this project, Random Forest is configured using:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

Unlike Logistic Regression, Random Forest does not use the standardized dataset in this implementation.

Random Forest is useful for this project because it can identify more complex relationships between lifestyle factors and hypertension status. It also provides built-in feature importance scores, which are displayed in the dashboard.

---

## 3. XGBoost

XGBoost is a gradient boosting algorithm that builds a sequence of decision trees, with later trees attempting to improve the model based on errors from previous trees.

When XGBoost is installed, the dashboard creates the model using:

```python
xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_estimators=100
)
```

XGBoost is optional in this project.

If the package is not installed, the dashboard continues using Logistic Regression and Random Forest.

To install XGBoost:

```bash
pip install xgboost
```

---

# 🧹 Features Used for Model Training

The machine learning models use the following demographic and lifestyle-related features:

```text
Age
Salt_Intake
Stress_Score
Sleep_Duration
BMI
Family_History
Exercise_Level
Smoking_Status
```

The target variable is:

```text
Has_Hypertension
```

The target is converted into binary values:

```text
No  → 0
Yes → 1
```

This allows the classification models to learn patterns associated with hypertensive and non-hypertensive records.

---

# 🛡️ Data Leakage Prevention

An important consideration in this project is preventing **data leakage**.

Data leakage occurs when information that should not be available to the predictive model indirectly or directly reveals information about the target.

The dashboard focuses on demographic and lifestyle predictors.

Therefore, even if the uploaded dataset contains:

```text
BP_History
Medication
```

these variables are not selected as model features.

The model feature set remains:

```python
lifestyle_features = [
    "Age",
    "Salt_Intake",
    "Stress_Score",
    "Sleep_Duration",
    "BMI",
    "Family_History",
    "Exercise_Level",
    "Smoking_Status"
]
```

This design allows the project to focus specifically on relationships between lifestyle or demographic characteristics and hypertension status.

---

# ⚙️ Data Preprocessing

Before model training, the uploaded dataset goes through several preprocessing steps.

## Numerical Variables

The following variables are converted into numerical data:

```text
Age
Salt_Intake
Stress_Score
Sleep_Duration
BMI
```

For example:

```python
df_processed["Age"] = pd.to_numeric(
    df_processed["Age"],
    errors="coerce"
)
```

The same approach is applied to the other numerical variables.

---

## Categorical Variables

The main categorical predictors are:

```text
Family_History
Exercise_Level
Smoking_Status
```

Because machine learning algorithms require numerical input, categorical variables are encoded using:

```python
LabelEncoder()
```

For example:

```python
le = LabelEncoder()
X[col] = le.fit_transform(X[col].astype(str))
```

The fitted encoders are stored so that the same encoding can later be applied when users enter information through the Risk Assessment page.

---

# 📏 Feature Scaling

Feature scaling is performed using:

```python
StandardScaler()
```

In the current implementation, scaling is specifically used for **Logistic Regression**.

For example:

```python
scaler = StandardScaler()

X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns
)
```

Random Forest and XGBoost use the unscaled feature representation.

---

# ✂️ Train-Test Split

The dataset is divided into training and testing portions using an **80:20 split**.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

This means:

```text
80% → Used for model training

20% → Used for final model evaluation
```

The split uses:

```python
stratify=y
```

to help maintain the distribution of the target classes across the training and test portions.

A fixed:

```python
random_state=42
```

is also used to make the randomized split reproducible when the dataset and environment remain the same.

---

# 🔄 Cross-Validation

In addition to the train-test split, the project uses **Stratified 5-Fold Cross-Validation**.

```python
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

The training data is divided into five folds.

Conceptually:

```text
                Training Dataset
                       │
                       ▼
               ┌───────────────┐
               │     Fold 1    │
               │     Fold 2    │
               │     Fold 3    │
               │     Fold 4    │
               │     Fold 5    │
               └───────────────┘
                       │
                       ▼
              Model Evaluation
                       │
                       ▼
             Mean CV Accuracy
```

For each model, cross-validation accuracy is calculated using:

```python
cross_val_score(
    model,
    curr_X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)
```

The dashboard records both:

```text
Mean Cross-Validation Accuracy
Cross-Validation Standard Deviation
```

---

# 🏆 Best Model Selection

After cross-validation, the dashboard compares the mean cross-validation accuracy of the trained models.

The best model is selected using:

```python
best_model_name = max(
    cv_scores_dict,
    key=cv_scores_dict.get
)
```

Therefore, the model with the **highest mean cross-validation accuracy** is selected as the best-performing model for the current dataset and selected age range.

The selected model is subsequently used in the machine-learning-based Risk Assessment page.

---

# 📊 Model Evaluation Metrics

The models are evaluated using several classification metrics.

The dashboard calculates:

```text
Accuracy
Precision
Recall
F1-Score
AUC-ROC
Cross-Validation Accuracy
Cross-Validation Standard Deviation
```

Using multiple metrics provides a more complete view of model performance than relying only on accuracy.

---

## Accuracy

Accuracy measures the proportion of predictions that were classified correctly.

```text
Accuracy = Correct Predictions / Total Predictions
```

The dashboard calculates it using:

```python
accuracy_score(y_test, y_pred)
```

---

## Precision

Precision measures how many records predicted as hypertensive were actually labelled as hypertensive.

```text
Precision = TP / (TP + FP)
```

It is calculated using:

```python
precision_score(
    y_test,
    y_pred,
    zero_division=0
)
```

---

## Recall

Recall measures how many of the actual hypertensive records were correctly identified by the model.

```text
Recall = TP / (TP + FN)
```

It is calculated using:

```python
recall_score(
    y_test,
    y_pred,
    zero_division=0
)
```

---

## F1-Score

The F1-Score combines Precision and Recall into a single measurement.

```text
F1 = 2 × (Precision × Recall)
       ─────────────────────
         Precision + Recall
```

The dashboard calculates this using:

```python
f1_score(
    y_test,
    y_pred,
    zero_division=0
)
```

---

## AUC-ROC

The dashboard also calculates the Area Under the Receiver Operating Characteristic Curve.

First, the predicted probability for the hypertension class is obtained:

```python
y_prob = model.predict_proba(
    curr_X_test
)[:, 1]
```

The AUC-ROC score is then calculated using:

```python
roc_auc_score(
    y_test,
    y_prob
)
```

This provides information about how effectively the model's predicted scores discriminate between the two target classes.

---

# 📉 Confusion Matrix

Each trained model also receives a confusion matrix.

The confusion matrix provides a detailed view of classification outcomes:

```text
                         Predicted
                    No              Yes

Actual No           TN               FP

Actual Yes          FN               TP
```

Where:

```text
TN = True Negative
FP = False Positive
FN = False Negative
TP = True Positive
```

The dashboard calculates this using:

```python
confusion_matrix(
    y_test,
    y_pred
)
```

and displays it as an interactive visualization.

---

# 📈 Learning Curves

The dashboard provides learning curves for each trained machine learning model.

Learning curves show how the model's performance changes as progressively more training data is used.

The application displays:

```text
Training Accuracy
Validation Accuracy
```

against:

```text
Number of Training Samples
```

The learning curves are generated using:

```python
learning_curve(
    model,
    X,
    y,
    cv=cv,
    train_sizes=train_sizes,
    scoring="accuracy",
    shuffle=True,
    random_state=42
)
```

The dashboard also displays ±1 standard deviation around the training and validation scores.

This visualization can help investigate the relationship between training-set size and model performance.

---

# ⭐ Feature Importance

Understanding which variables influence a model is an important part of this project.

The dashboard therefore provides feature importance analysis for each supported model.

## Logistic Regression

For Logistic Regression, the absolute values of the fitted coefficients are used:

```python
np.abs(
    model.coef_[0]
)
```

Larger absolute coefficients indicate features with stronger influence within the fitted Logistic Regression model after its preprocessing.

---

## Random Forest

Random Forest provides feature importance through:

```python
model.feature_importances_
```

The resulting values are associated with the model features and displayed as a bar chart.

---

## XGBoost

When XGBoost is available, its:

```python
model.feature_importances_
```

values are also displayed.

Since the three algorithms work differently, the importance ranking does not necessarily need to be identical between models.

---

# 🧠 SHAP Analysis

The project also implements **SHAP (SHapley Additive exPlanations)** to provide additional model interpretability.

SHAP values are used to investigate how features contribute to model predictions.

Instead of only answering:

```text
"What did the model predict?"
```

SHAP analysis helps investigate:

```text
"Which features influenced the model's predictions?"
```

The dashboard provides SHAP analysis for:

```text
Logistic Regression
Random Forest
XGBoost
```

when the required libraries and model results are available.

---

## Logistic Regression SHAP

For Logistic Regression, the application uses:

```python
shap.LinearExplainer()
```

For example:

```python
explainer = shap.LinearExplainer(
    model,
    curr_X_train,
    feature_names=feature_names
)
```

The SHAP values are calculated against the test samples:

```python
shap_values = explainer.shap_values(
    curr_X_test
)
```

---

## Random Forest SHAP

Random Forest uses:

```python
shap.TreeExplainer()
```

A sample of the training data is used as background data:

```python
background = shap.sample(
    curr_X_train,
    min(100, len(curr_X_train))
)
```

The explainer is then created:

```python
explainer = shap.TreeExplainer(
    model,
    data=background,
    feature_perturbation="interventional"
)
```

---

## XGBoost SHAP

XGBoost also uses:

```python
shap.TreeExplainer()
```

when both XGBoost and SHAP are available.

This enables the dashboard to investigate feature contributions within the trained boosting model.

---

# 📊 SHAP Visualizations

Two main SHAP visualizations are provided.

## SHAP Summary / Beeswarm Plot

The summary plot displays the distribution of SHAP values across test samples.

Conceptually:

```text
Feature Value
      │
      ▼
Model Prediction
      │
      ▼
SHAP Contribution
      │
      ├── Positive contribution
      │
      └── Negative contribution
```

The visualization helps users inspect how different feature values are associated with changes in model output.

---

## Mean Absolute SHAP Values

The dashboard also generates a bar plot using mean absolute SHAP values.

This provides a global indication of how strongly each variable influences predictions across the analyzed samples.

Higher mean absolute SHAP values represent features with greater overall influence on the model output.

> **Important:** SHAP explains how the trained model behaves. It does not prove that a feature medically causes hypertension.

---

# 🎯 Risk Assessment

After the machine learning models have been trained, the dashboard provides a personalized Risk Assessment interface.

Users can enter:

```text
Age
BMI
Family History
Exercise Level
Smoking Status
Daily Salt Intake
Stress Score
Sleep Duration
```

These inputs correspond to the same predictors used during model training.

The application prepares the user input and applies the stored categorical encoders.

If Logistic Regression is selected as the best model, the stored scaler is also applied before prediction.

---

# 🔮 ML-Based Prediction

The selected best model calculates:

```python
model.predict_proba(...)
```

The probability associated with the hypertension class is converted into a percentage:

```python
prediction_proba[1] * 100
```

For example:

```text
Predicted Probability = 0.64

Displayed Risk Score = 64 / 100
```

The resulting score represents the model's estimated probability for the positive class based on patterns learned from the uploaded dataset.

It should **not** be interpreted as a medical diagnosis.

---

# 🚦 Displayed Risk Categories

For easier interpretation within the dashboard, the resulting score is grouped into four application-defined categories:

```text
0 - 29       → Low Risk
30 - 49      → Moderate Risk
50 - 69      → High Risk
70 - 100     → Very High Risk
```

These categories are used for dashboard presentation purposes.

> They are not presented as clinically validated hypertension diagnostic thresholds.

---

# 🔄 Simplified Scoring Fallback

If the machine learning models cannot be trained or an ML prediction cannot be generated, the application can fall back to a simplified rule-based scoring function.

The fallback considers:

```text
Age
BMI
Family History
Exercise
Smoking
Salt Intake
Stress
Sleep Duration
```

This fallback allows the dashboard demonstration to continue when an ML model is unavailable.

However, the simplified score should also be treated as an educational application feature and **not as a validated clinical hypertension risk score**.
