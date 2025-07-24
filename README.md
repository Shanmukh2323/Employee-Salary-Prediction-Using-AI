# Employee-Salary-Prediction-Using-AI

This project predicts employee salaries based on various features like job title, education, experience, and more using a Random Forest Regression model. It includes a Streamlit web app for user interaction and visualization.

---

## 📌 Project Overview

- 🎯 **Goal:** Predict employee salary based on input attributes.
- 🧠 **Model:** RandomForestRegressor (non-linear and high-performance)
- 🌐 **Interface:** Built with Streamlit for easy user interaction

---

## 📁 Project Structure
employee-salary-prediction/
│
├── data/ # Dataset files (raw & cleaned)
│ └── employee_data.csv
│
├── model/ # Training and evaluation code
│ └── train_model.py
│ └── model.pkl
│
├── app/ # Streamlit web app
│ └── app.py
│
├── notebooks/ # EDA and model experiments
│ └── SalaryPredictionEDA.ipynb
│
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## 🔧 Features & Workflow

- ✅ Cleaned and preprocessed dataset
- ✅ One-Hot Encoding for categorical features
- ✅ Normalization for numerical columns
- ✅ Trained a Random Forest Regressor
- ✅ Evaluated model using R² Score, MAE
- ✅ Feature importance visualization
- ✅ Streamlit app for:
  - Interactive user input
  - Real-time salary prediction
  - Visualizations (salary distribution, top features)

---

<img width="1920" height="2691" alt="screencapture-localhost-8501-2025-07-24-20_45_24" src="https://github.com/user-attachments/assets/93c61135-d661-4997-b912-6619eef7a160" />
<img width="618" height="368" alt="Screenshot 2025-07-24 193420" src="https://github.com/user-attachments/assets/fe59bc20-330e-4a5c-8cef-eb35ed231078" />

## 📊 Libraries & Tools

- Python (Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib)
- Streamlit
- Jupyter Notebook
- VS Code or Jupyter Lab

🚀 How to Run the App Locally

1. **Clone the Repository:**
   bash
   git clone https://github.com/Shanmukh2323/Employee-Salary-Prediction-Using-AI.git
   cd employee-salary-prediction
2. Install Dependencies:
   bash
pip install -r requirements.txt

3. Run the Streamlit App:
   bash
streamlit run app/app.py

📈 Model Performance
R² Score: ~0.85

MAE: ~3000 (can vary by dataset)

Random Forest provided better performance over linear models for complex data relationships.

📬 Contact
Author: [Shanmukh Sahukari]
LinkedIn: https://www.linkedin.com/in/shanmukh-sahukari/
Email: shanmukhsahukari1234@gmail.com
