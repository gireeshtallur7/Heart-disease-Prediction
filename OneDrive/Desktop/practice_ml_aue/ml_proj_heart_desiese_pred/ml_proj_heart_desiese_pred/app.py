import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Prediction")
st.title("❤️ Heart Disease Prediction App")

st.write("App started...")  # Debug line

model = None
scaler = None
expected_cols = None

try:
    model = joblib.load("logistic_heart_model.pkl")
    scaler = joblib.load("heart_scaler.pkl")
    expected_cols = joblib.load("heart_Columns.pkl")
    st.success("Model loaded successfully ✅")
except Exception as e:
    st.error("❌ ERROR LOADING MODEL:")
    st.write(e)

st.markdown("provide the following details")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("SEX", ["Male", "Female"])
chest_pain_type = st.selectbox("Chest_pain_type", ["ATA", "NAP", "TA", "ASY"])
Resting_BP = st.number_input("Resting_BP (mm HG)", 88, 200, 120)
colestrol = st.number_input("colestrol_(mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting_blood_suger >120 mg/dl", [0, 1])
resting_ecg = st.selectbox("Resting_ecg", ["Normal", "ST", "LVH"])
max_heart_rate = st.slider("Max_heart_rate", 60, 220, 150)
exercise_angina = st.selectbox("Excersise_Induced_angina", ["Y", "N"])
old_peak = st.slider("Old_peak (ST Depression)", 0.0, 6.0, 1.0, 0.1)
st_slope = st.selectbox("st_slope", ["up", "flat", "Down"])

if st.button("Predict"):

    # Safety check
    if "model" not in locals():
        st.error("Model not loaded properly ❌")
    else:
        raw_input = {
            "Age": age,
            "RestingBP": Resting_BP,
            "Cholesterol": colestrol,
            "FastingBS": fasting_bs,
            "MaxHR": max_heart_rate,
            "Oldpeak": old_peak
        }

        if sex == "Male":
            raw_input["Sex_Male"] = 1
            raw_input["Sex_Female"] = 0
        else:
            raw_input["Sex_Male"] = 0
            raw_input["Sex_Female"] = 1

        raw_input[f"ChestPainType_{chest_pain_type}"] = 1
        raw_input[f"RestingECG_{resting_ecg}"] = 1
        raw_input[f"ExerciseAngina_{exercise_angina}"] = 1
        raw_input[f"ST_Slope_{st_slope}"] = 1

        input_dataframe = pd.DataFrame([raw_input])

        for col in expected_cols:
            if col not in input_dataframe.columns:
                input_dataframe[col] = 0

        input_dataframe = input_dataframe[expected_cols]

        scaled_input = scaler.transform(input_dataframe)
        prediction = model.predict(scaled_input)[0]

        if prediction == 1:
            st.error("High Risk of Heart Disease")
        else:
            st.success("Low Risk of Heart Disease")