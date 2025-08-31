# Heart Disease Prediction ML App

This project is a **Heart Disease Prediction application** built using Machine Learning.  
Users can input health parameters such as age, cholesterol level, blood pressure, etc., and the app predicts the likelihood of heart disease.

## Features
- Predict heart disease risk using user input data.
- Built with Python, Pandas, Scikit-learn, and Streamlit.
- Provides simple and interactive web interface.

## Project Structure
heart-disease-ml/
│
├── heart_disease_app.py # Main Streamlit app
├── train_model.py # Script to train ML model
├── model.pkl # Trained ML model
├── heart.csv # Dataset
├── requirements.txt # Required Python libraries
├── .gitignore # Ignore unnecessary files
└── README.md # Project documentation

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Fowzi225577/heart-disease-prediction-ML.git
cd heart-disease-ml
Create a virtual environment (optional but recommended):
python -m venv .venv
Install required libraries:

pip install -r requirements.txt
Run the Streamlit app:

streamlit run heart_disease_app.py
Train the Model (Optional)

To retrain the model using the dataset:

python train_model.py
