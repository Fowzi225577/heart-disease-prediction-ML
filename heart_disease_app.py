# heart_disease_app.py
# Run with: streamlit run heart_disease_app.py
# heart_disease_app.py
# Run with: streamlit run heart_disease_app.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("heart.csv")  # Make sure heart.csv is in same folder
    except FileNotFoundError:
        st.error("❌ heart.csv not found. Download from Kaggle and place it in this folder.")
        st.stop()
    
    # One-hot encode categorical columns
    categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    return data

df = load_data()

# Features and target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Save feature names to align user input later
feature_names = X.columns.tolist()

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.title("❤️ Heart Disease Prediction App")
st.write(f"Model Accuracy on Test Set: **{acc:.2f}**")

# ---------------------
# User Input Form
# ---------------------
st.header("Enter Patient Data")

def user_input():
    Age = st.slider("Age", 20, 100, 50)
    RestingBP = st.slider("Resting Blood Pressure", 80, 200, 120)
    Cholesterol = st.slider("Cholesterol", 100, 400, 200)
    FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    MaxHR = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    Oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)

    Sex = st.selectbox("Sex", ["M", "F"])
    ChestPainType = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
    RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    ExerciseAngina = st.selectbox("Exercise Angina", ["N", "Y"])
    ST_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    # Create dictionary for one-hot encoding
    data = {
        "Age": Age,
        "RestingBP": RestingBP,
        "Cholesterol": Cholesterol,
        "FastingBS": FastingBS,
        "MaxHR": MaxHR,
        "Oldpeak": Oldpeak,
        "Sex_M": 1 if Sex=="M" else 0,
        "ChestPainType_ATA": 1 if ChestPainType=="ATA" else 0,
        "ChestPainType_NAP": 1 if ChestPainType=="NAP" else 0,
        "ChestPainType_ASY": 1 if ChestPainType=="ASY" else 0,
        "RestingECG_ST": 1 if RestingECG=="ST" else 0,
        "RestingECG_LVH": 1 if RestingECG=="LVH" else 0,
        "ExerciseAngina_Y": 1 if ExerciseAngina=="Y" else 0,
        "ST_Slope_Flat": 1 if ST_Slope=="Flat" else 0,
        "ST_Slope_Down": 1 if ST_Slope=="Down" else 0
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([data])

    # Align columns with training data (fill missing with 0)
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    return input_df

input_df = user_input()

# ---------------------
# Prediction
# ---------------------
if st.button("Predict"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0, 1]

    st.subheader("Prediction Result:")
    st.write("⚠️ High Risk of Heart Disease" if prediction==1 else "✅ Likely NO Heart Disease")
    st.write(f"Predicted Probability: **{probability:.2%}**")
