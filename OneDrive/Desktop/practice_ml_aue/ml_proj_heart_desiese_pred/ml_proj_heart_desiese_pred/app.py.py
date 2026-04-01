 
import streamlit as st
import pandas as pd
import  joblib
model=joblib.load("logistic_heart_model.pkl")
scaler=joblib.load("heart_scaler.pkl") 
expected_cols=joblib.load("heart_Columns.pkl")
st.title("Heart stroke prediction  by Gireesh")
st.markdown("provide the following details")
age=st.slider("Age",18,100,40)
sex=st.selectbox("SEX" ,["Male","Female"])
chest_pain_type=st.selectbox("Chest_pain_type",["ATA","NAP","TA","ASY"])
Resting_BP=st.number_input("Resting_BP (mm HG)",88,200,120)
colestrol=st.number_input("colestrol_(mg/dL)",100,600,200)
fasting_bs=st.selectbox("Fasting_blood_suger >120 mg/dl",[0,1])
resting_ecg=st.selectbox("Resting_ecg",["Normal","ST","LVH"])
max_heart_rate=st.slider("Max_heart_rate",60,220,150)
exercise_angina=st.selectbox("Excersise_Induced_angina",["Y","N"])
old_peak=st.slider("Old_peak (ST Depression)",0.0,6.0,1.0,0.1)
st_slope=st.selectbox("st_slope",["up","flat","Down"])
if  st.button("Predict"):
    raw_input={        
    "Age":age,
    "RestingBP":Resting_BP,
    "Cholesterol":colestrol,
    "FastingBS":fasting_bs,
    "MaxHR":max_heart_rate,
    "Oldpeak":old_peak
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
    
    input_dataframe=pd.DataFrame([raw_input])
    for col in expected_cols:
        if col not in input_dataframe.columns:
            input_dataframe[col]=0
    input_dataframe=input_dataframe[expected_cols] #now my input data frame is ready    
    scaled_input=scaler.transform(input_dataframe) #i have scleaed all the features and in all model output is came from scaled only  it's not badd to use the  scaled feature ,but it is bad to not use the scaled features in knn  or svm 
    prediction=model.predict(scaled_input)[0]
    if prediction==1:
        st.error("High  Risk of Heart Disease")
    else :
        st.success(" Low  risk of  heart disease")
