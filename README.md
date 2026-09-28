# 🫀 Hypertension Risk Predictor Dashboard

An interactive machine learning dashboard for exploring hypertension risk factors, comparing predictive models, interpreting model behavior, and estimating hypertension risk from lifestyle and demographic information.

Built with Python and Streamlit.

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
