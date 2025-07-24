import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = pickle.load(open('model/model.pkl', 'rb'))

st.title("💼 Employee Income Prediction App")
st.write("Enter new employee details below:")

# Input form
with st.form("employee_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Federal-gov'])
    educational_num = st.slider("Education Level (numeric)", 1, 16, 10)
    marital_status = st.selectbox("Marital Status", ['Never-married', 'Married-civ-spouse', 'Divorced', 'Separated', 'Widowed'])
    occupation = st.selectbox("Occupation", ['Tech-support', 'Sales', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Craft-repair'])
    relationship = st.selectbox("Relationship", ['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried'])
    race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo'])
    gender = st.selectbox("gender", ['Male', 'Female'])
    hours_per_week = st.slider("Hours per Week", 1, 100, 40)
    capital_gain = st.number_input("Capital Gain", value=0)
    capital_loss = st.number_input("Capital Loss", value=0)

    submitted = st.form_submit_button("Predict Income")

if submitted:
    input_data = pd.DataFrame([{
        'age': age,
        'workclass': workclass,
        'educational-num': educational_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'hours-per-week': hours_per_week,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss
    }])

    # Prediction
    prediction = model.predict(input_data)[0]
    st.subheader("💰 Predicted Income Category")
    st.success(f"The employee is predicted to earn: **{prediction}**")

    # 🔍 Visualization
    fig, ax = plt.subplots()
    sns.barplot(x=['<=50K', '>50K'], y=[int(prediction == '<=50K'), int(prediction == '>50K')], palette="viridis", ax=ax)
    ax.set_title("Predicted Income Distribution")
    ax.set_ylabel("Probability (1 = True)")
    st.pyplot(fig)
