# 🧠 Insurance Premium Prediction using Machine Learning
link - https://ml-premium-prediction-model.streamlit.app
This project predicts insurance premiums using customer demographic and income data. After experimenting with various models, **XGBoost** was selected as the final model due to its superior accuracy and performance.

---

## 📁 Project Structure

premium-prediction-ml/
├── artifacts/
│ ├── model_young.joblib
│ ├── model_senior.joblib
│ └── scaler_young.joblib
├── prediction_helper.py
├── main.py
├── requirements.txt
└── README.md

## 📌 Problem Statement

Insurance companies aim to offer accurate premium pricing based on various customer attributes such as age, income, and health status. This machine learning model helps predict insurance premiums for new customers using historical data and modern ML techniques.

---

## 🚀 Features

- 🔍 Preprocessing and feature scaling using `MinMaxScaler`
- 🧠 Trained on multiple algorithms — best model selected as **XGBoost**
- 📦 Model artifacts saved in `.joblib` format for fast loading
- 🧪 `prediction_helper.py` script loads model & scaler for predictions
- 🖥️ Ready to plug into an API or frontend

---

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/premium-prediction-ml.git
   cd premium-prediction-ml


📊 Model Training (Optional - if retraining)
# Example
print(predict([[25, 1, 50000]]))  # Age, Gender, Income (example input)


from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib

# Train scaler and model
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)

model = XGBRegressor()
model.fit(X_scaled, y_train)

# Save artifacts
joblib.dump(model, "artifacts/model_young.joblib")
joblib.dump(scaler, "artifacts/scaler_young.joblib")

📦 Dependencies

Python 3.11+
pandas
numpy
scikit-learn
xgboost
joblib
Install via pip install -r requirements.txt

📄 License

Apache License. Feel free to use, modify, and distribute.

To generate this automatically from your environment:
```bash
pip freeze > requirements.txt
