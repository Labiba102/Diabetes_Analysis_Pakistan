# Diabetes Prediction Using Logistic Regression on Pakistani Dataset

This project performs data preprocessing, feature engineering, and classification modeling using logistic regression on a Pakistani diabetes dataset. The aim is to build a predictive model to classify individuals as diabetic or non-diabetic based on clinical and demographic features.

## Dataset

* **Source**: [Pakistani Diabetes Dataset on Kaggle](https://www.kaggle.com/datasets/mshoaibishaaq/pakistani-diabetes-dataset)
* **Size**: 912 rows × 19 columns
* **Features** include:

  * Demographics (e.g., Age, Gender, Region)
  * Health metrics (e.g., Weight, BMI, Blood Pressure, A1c)
  * Diabetes-related symptoms and lab indicators
  * Target column: `Diabetes_Outcome` (0 = Non-diabetic, 1 = Diabetic)

## Tools & Libraries

* Python (pandas, numpy, seaborn, matplotlib)
* Machine Learning: `scikit-learn`
* Data Source Access: `kagglehub`

## Workflow

### 1. **Data Loading**

Used `kagglehub` to load the dataset directly from Kaggle.

```python
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "mshoaibishaaq/pakistani-diabetes-dataset",
  path=file_path)
```

### 2. **Data Cleaning & Preprocessing**

* Removed duplicates and unrealistic values
* Renamed columns for better readability
* Verified and handled null values (none present)
* Normalized numerical features
* Label-encoded categorical features

### 3. **Feature Engineering**

* Cleaned outliers using domain knowledge thresholds (e.g., valid BMI range: 10–60)
* Verified class distribution (balanced)

### 4. **Modeling**

* Applied `LogisticRegressionCV` with 5-fold cross-validation and `f1` score as evaluation metric
* Used `class_weight='balanced'` to address any imbalance
* Trained model on 70% of data, tested on 30%

```python
model = LogisticRegressionCV(
    Cs=10,
    cv=5,
    penalty='l2',
    solver='liblinear',
    scoring='f1',
    class_weight='balanced',
    random_state=42,
    max_iter=1000
)
```

### 5. **Evaluation**

* **ROC AUC Score**: `1.0`
* **Confusion Matrix**: Perfect classification
* **Classification Report**: Visualized using seaborn heatmap

## Visualizations

* Heatmaps for confusion matrix and classification report
* Distribution checks via `.describe()` and `value_counts()`

## Installation & Requirements

Install dependencies with:

```bash
pip install kagglehub[pandas-datasets] pandas seaborn matplotlib scikit-learn
```

Make sure your environment includes:

* Python 3.7+
* `kagglehub >= 0.3.10`

## Future Improvements

* Add hyperparameter tuning with `GridSearchCV`
* Experiment with ensemble models (Random Forest, XGBoost)
* Evaluate on external datasets
* Deploy the model using Streamlit or Flask

## Project Structure

```
.
├── diabetes_prediction.ipynb
├── Pakistani_Diabetes_Dataset.csv
├── README.md
└── requirements.txt (optional)
```

## Author

Labiba Shahab
[LinkedIn](https://www.linkedin.com/in/labiba-shahab-2ba3261b1/)
