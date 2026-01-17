# 🩺 Disease Symptom Prediction using Machine Learning

## 📌 Project Overview

This project builds an **end-to-end Machine Learning pipeline** to predict diseases based on patient symptoms. The dataset contains multiple symptom columns (`Symptom_1` to `Symptom_17`) with variable-length symptom information per patient.

The solution follows **industry best practices** using:

* `ColumnTransformer`
* `Pipeline`
* `GridSearchCV`
* Multiple classification models
* Proper evaluation metrics

The goal is to create a **scalable, leakage-free, and reproducible ML workflow**.

---

## 📂 Dataset Description

* **Target Column:** `Disease`
* **Feature Columns:** `Symptom_1` to `Symptom_17`
* Symptoms are categorical values
* Missing values (`NaN`) indicate **symptom not present**
* Dataset contains **duplicate rows**, which are removed during preprocessing

---

## 🧹 Data Preprocessing

### Steps Performed:

1. **Removed duplicate rows** to avoid data leakage
2. **Separated features and target**
3. **Encoded target variable** using `LabelEncoder`
4. Used **OneHotEncoder** for categorical symptom columns
5. All preprocessing handled inside a **ColumnTransformer**

```python
ColumnTransformer([
    ("symptoms", OneHotEncoder(handle_unknown="ignore"), symptom_columns)
])
```

`set_output(transform="pandas")` is used to retain DataFrame output after transformation.

---

## ⚙️ Machine Learning Pipeline

A unified **Pipeline** is created for each model:

```python
Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])
```

This ensures:

* No data leakage
* Clean cross-validation
* Easy hyperparameter tuning

---

## 🤖 Models Used

The following classification algorithms were trained and tuned using `GridSearchCV`:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)

Each model includes a **hyperparameter grid** optimized using 5-fold cross-validation.

---

## 📊 Model Evaluation Metrics

All models are evaluated on the test set using:

* Accuracy
* Precision (weighted)
* Recall (weighted)
* F1-score (weighted)
* ROC-AUC (One-vs-Rest for multiclass)

Results are stored in a DataFrame and sorted by **F1-score** for fair comparison.

---

## 🏆 Key Highlights

* ✔ End-to-end ML pipeline
* ✔ Handles missing symptoms correctly
* ✔ Avoids data leakage
* ✔ Scalable to new symptoms
* ✔ Interview & production ready

---

## 📁 Project Structure

```
Disease_Symptom_Prediction.ipynb
dataset.csv
requirements.txt
README.md
```

---

## 🚀 How to Run

1. Open the notebook: `Disease_Symptom_Prediction.ipynb`
2. Run all cells sequentially
3. View model comparison results at the end

---

## 🧠 Interview Explanation (One-liner)

> "I used a ColumnTransformer with OneHotEncoding inside a Pipeline and applied GridSearchCV across multiple classifiers to build a leakage-free, scalable disease prediction system."

---

## 📌 Future Improvements

* Add XGBoost / LightGBM
* Deploy using Streamlit or Flask
* Add symptom-based user input interface
* Save best model using `joblib`

---

✅ **Author:** Devendra Kushwah
