
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Page config
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

st.title("Diabetes Risk Prediction")
st.write("Enter patient clinical information:")

# Input fields
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose Level", 50, 200, 120)
    blood_pressure = st.number_input("Blood Pressure", 40, 130, 70)
    skin_thickness = st.number_input("Skin Thickness", 10, 100, 20)

with col2:
    insulin = st.number_input("Insulin Level", 0, 900, 80)
    bmi = st.number_input("BMI", 15.0, 60.0, 25.0)
    diabetes_pedigree = st.number_input("Diabetes Pedigree", 0.0, 2.5, 0.5)
    age = st.number_input("Age", 20, 100, 25)

if st.button("Predict Diabetes Risk"):
    try:
        # Load the diabetes dataset and train model directly in Streamlit
        # This avoids pickle compatibility issues
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        # Load data
        df = pd.read_csv('diabetes.csv')
        
        # Preprocess
        medical_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in medical_cols:
            df[col] = df[col].replace(0, df[col].mean())
        
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Train model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_scaled, y)
        
        # Prepare input
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [diabetes_pedigree],
            'Age': [age]
        })
        
        # Predict
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0, 1]
        
        # Show results
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("HIGH RISK OF DIABETES")
            st.write(f"Probability: {probability:.1%}")
        else:
            st.success("LOW RISK OF DIABETES")
            st.write(f"Probability: {probability:.1%}")
        
        st.progress(float(probability))
        
    except Exception as e:
        st.error(f"Please make sure diabetes.csv is in the same folder. Error: {str(e)}")

st.sidebar.info("For educational purposes only")
