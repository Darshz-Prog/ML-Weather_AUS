# ML-Weather_AUS

# 🌦️ Weather Prediction Using Machine Learning

This project predicts whether it will rain tomorrow based on historical weather data from various locations in Australia. The dataset is publicly available from [Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package).

---

## 📂 Dataset

* **Source**: [Kaggle - Weather Dataset (Rattle Package)](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)
* **File**: `weatherAUS.csv`
* **Target Column**: `RainTomorrow` (Yes/No)

This dataset includes features like temperature, humidity, wind direction, rainfall, pressure, and other meteorological variables from multiple weather stations across Australia.

---

## 🎯 Objective

Build a classification model to predict whether it will rain **tomorrow**, using the weather conditions observed **today**.

---

## 🛠️ Tools & Libraries

* Python
* Pandas & NumPy
* Scikit-learn
* XGBoost
* Matplotlib & Seaborn
* Joblib

---

## 🔍 Features Used

A sample of the features used for training:

* `MinTemp`, `MaxTemp`, `Rainfall`
* `WindGustSpeed`, `WindDir9am`, `WindDir3pm`
* `Humidity3pm`, `Pressure9am`, `Pressure3pm`
* `Cloud9am`, `Cloud3pm`
* `Location` (Categorical)
* And others...

---

## ⚙️ Preprocessing Steps

1. **Remove rows with missing target (`RainTomorrow`)**
2. **Drop non-useful columns** (e.g., `Date`)
3. **Handle missing values**

   * Optionally fill with mean/median or drop rows
4. **Encode categorical features**

   * Using `OneHotEncoder` for columns like `Location`, `WindGustDir`, etc.
5. **Feature Scaling**

   * `StandardScaler` for numerical features
6. **Label Encoding**

   * Convert `RainTomorrow`: `'Yes' → 1'`, `'No' → 0`

---

## 🤖 Model Training

We trained two classifiers:

### 1. `RandomForestClassifier`

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
```
---

## 📊 Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix
```

* **Confusion Matrix** plotted using `seaborn.heatmap()`
* **Classification Report** includes:

  * Precision
  * Recall
  * F1-Score
  * Support

---

## 📂 Model Saving

The trained pipeline (preprocessing + model) was saved using:

```python
import joblib
joblib.dump(model, 'AUS_Model.joblib')
```

Load it later:

```python
AUS = joblib.load('Aus_MOdel.joblib')
y_pred = AUS.predict(X_test)
```

---

## 📈 Future Improvements

* Hyperparameter tuning using `GridSearchCV`
* Feature engineering (e.g., time-based season features)
* Imputation strategies for missing data
* Model interpretability (e.g., SHAP values)

---

## 📌 Project Structure

```
weather-ml/
├── weatherAUS.csv
├── notebook.ipynb
├── rain_prediction_pipeline.joblib
└── README.md
```

---

## 🧠 Author

This project was implemented and documented by **Darsh**.

---

## 📜 License

This project is for educational use. Dataset licensed under Kaggle’s terms.
