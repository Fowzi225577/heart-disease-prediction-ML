
# Heart Disease Prediction using Machine Learning

## Project Overview
This project is a **Heart Disease Prediction system** that uses machine learning to predict the likelihood of a person having heart disease based on medical features. The model is trained on the popular **Heart Disease Dataset** and provides a simple web interface for user input using **Streamlit**.

## Features
- Predicts heart disease presence using medical parameters
- Web-based interface for easy input and prediction
- Uses a trained machine learning model (`model.pkl`)
- Data processing and feature selection included

## Project Structure
heart-disease-ml/
│
├── heart_disease_app.py # Main Streamlit app
├── train_model.py # Script to train and save the ML model
├── heart.csv # Dataset
├── model.pkl # Trained machine learning model
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Files/folders to ignore in Git

bash
Copy code

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/Fowzi225577/heart-disease-prediction-ML.git
cd heart-disease-prediction-ML
Create a virtual environment

bash
Copy code
python -m venv .venv
Activate the virtual environment

Windows:

bash
Copy code
.venv\Scripts\activate
Mac/Linux:

bash
Copy code
source .venv/bin/activate
Install dependencies

bash
Copy code
pip install -r requirements.txt
How to Run
Make sure the virtual environment is activated

Run the Streamlit app

bash
Copy code
streamlit run heart_disease_app.py
Open the provided local URL in your browser and input patient details to get predictions.

How to Train the Model
If you want to retrain the model:

bash
Copy code
python train_model.py
This will generate a new model.pkl file which the app uses for predictions.

Dependencies
Python 3.x

pandas

scikit-learn

streamlit

joblib

Install all dependencies using:

bash
Copy code
pip install pandas scikit-learn streamlit joblib
