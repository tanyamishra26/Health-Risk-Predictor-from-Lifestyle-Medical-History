import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load your ML pipeline model
pipe = joblib.load("lightgbm_risk_model.pkl")

# Title
st.set_page_config(page_title="Health Risk Predictor", layout="wide")
st.title("💉 Health Risk Prediction System")
st.write("Predict your *percentage risk* for 5 major health conditions using lifestyle & medical data.")

# Sidebar Input Form
st.sidebar.header("User Input Panel")
st.sidebar.write("Fill the details below:")

# Side-by-side input layout
col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", 1, 120, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    ethnicity = st.selectbox("Ethnicity", ["Asian", "Black", "White", "Hispanic", "Other"])
    education_level = st.selectbox("Education Level", ["No formal", "Highschool", "Graduate", "Postgraduate"])
    income_level = st.selectbox("Income Level", ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"])
    employment_status = st.selectbox("Employment Status", ["Unemployed", "Employed"])

    smoking_status = st.selectbox("Smoking Status", ["Smoker", "Non-smoker"])
    family_history_diabetes = st.selectbox("Family History of Diabetes", ["Yes", "No"])
    hypertension_history = st.selectbox("Hypertension History", ["Yes", "No"])
    cardiovascular_history = st.selectbox("Cardiovascular History", ["Yes", "No"])

with col2:
    bmi = st.number_input("BMI", 10.0, 60.0, 24.5)
    systolic_bp = st.number_input("Systolic BP", 80.0, 200.0, 120.0)
    diastolic_bp = st.number_input("Diastolic BP", 40.0, 140.0, 80.0)

    cholesterol_total = st.number_input("Total Cholesterol", 100.0, 400.0, 180.0)
    hdl_cholesterol = st.number_input("HDL Cholesterol", 20.0, 100.0, 55.0)
    ldl_cholesterol = st.number_input("LDL Cholesterol", 30.0, 250.0, 120.0)
    triglycerides = st.number_input("Triglycerides", 30.0, 500.0, 140.0)

    glucose_fasting = st.number_input("Fasting Glucose", 50.0, 300.0, 95.0)
    glucose_postprandial = st.number_input("Post-Meal Glucose", 70.0, 350.0, 140.0)
    insulin_level = st.number_input("Insulin Level", 1.0, 50.0, 10.0)

    hba1c = st.number_input("HbA1c (%)", 3.0, 12.0, 5.5)

physical_activity_minutes_per_week = st.slider("Physical Activity (min/week)", 0, 600, 150)
diet_score = st.slider("Diet Quality Score", 0, 100, 60)
sleep_hours_per_day = st.slider("Sleep Hours per Day", 3, 12, 7)
screen_time_hours_per_day = st.slider("Screen Time (hours/day)", 0, 16, 4)

heart_rate = st.slider("Heart Rate (bpm)", 40, 150, 72)
alcohol_consumption_per_week = st.slider("Alcohol Consumption (units/week)", 0, 30, 0)
waist_to_hip_ratio = st.number_input("Waist-to-Hip Ratio", 0.6, 1.5, 0.9)

# Prediction Button
if st.button("🔍 Predict My Health Risks"):
    
    input_data = {
        "Age": Age,
        "gender": gender,
        "ethnicity": ethnicity,
        "education_level": education_level,
        "income_level": income_level,
        "employment_status": employment_status,
        "smoking_status": smoking_status,
        "physical_activity_minutes_per_week": physical_activity_minutes_per_week,
        "diet_score": diet_score,
        "sleep_hours_per_day": sleep_hours_per_day,
        "screen_time_hours_per_day": screen_time_hours_per_day,
        "family_history_diabetes": family_history_diabetes,
        "hypertension_history": hypertension_history,
        "cardiovascular_history": cardiovascular_history,
        "bmi": bmi,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "cholesterol_total": cholesterol_total,
        "hdl_cholesterol": hdl_cholesterol,
        "ldl_cholesterol": ldl_cholesterol,
        "triglycerides": triglycerides,
        "glucose_fasting": glucose_fasting,
        "glucose_postprandial": glucose_postprandial,
        "insulin_level": insulin_level,
        "hba1c": hba1c,
        "heart_rate": heart_rate,
        "alcohol_consumption_per_week": alcohol_consumption_per_week,
        "waist_to_hip_ratio": waist_to_hip_ratio
    }

    df = pd.DataFrame([input_data])

    # Convert Yes/No binary
    for col in ["family_history_diabetes", "hypertension_history", "cardiovascular_history"]:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    preds = pipe.predict(df)[0]

    st.subheader("📊 Predicted Health Risks (%):")
    labels = ["Diabetes", "Hypertension", "Heart Disease", "Obesity", "Cholesterol"]

    for i, risk in enumerate(labels):
        st.write(f"**{risk} Risk:** {round(float(preds[i]), 2)}%")

    # Optional graphical bars
    st.subheader("📉 Risk Level Visualization")
    st.bar_chart(pd.DataFrame({"Risk (%)": preds}, index=labels))
