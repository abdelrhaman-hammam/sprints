import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

@st.cache_resource
def load_deployment_package():
    try:
        
        deployment_path = "C:/studies/sprints/Heart_Disease_Project/deployment/model_package.pkl"
        scaler_path = "C:/studies/sprints/Heart_Disease_Project/models/scaler.pkl"
        pca_path = "C:/studies/sprints/Heart_Disease_Project/models/pca_model.pkl"
        
        
        if os.path.exists(deployment_path):
            deployment_package = joblib.load(deployment_path)
            model = deployment_package['model']
            feature_names = deployment_package['feature_names']
            model_info = deployment_package
            
            
            scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
            
            
            pca = joblib.load(pca_path) if os.path.exists(pca_path) else None
            
            st.sidebar.success(" Deployment package loaded successfully!")
            return model, scaler, pca, feature_names, model_info
            
        
        elif os.path.exists("C:/studies/sprints/Heart_Disease_Project/models/final_model.pkl") and os.path.exists(scaler_path):
            model = joblib.load("C:/studies/sprints/Heart_Disease_Project/models/final_model.pkl")
            scaler = joblib.load(scaler_path)
            
            
            try:
                feature_names = list(model.feature_names_in_)
            except:
                feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                               'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            
            pca = joblib.load(pca_path) if os.path.exists(pca_path) else None
            model_info = {'model_type': 'RandomForest', 'version': '1.0'}
            
            st.sidebar.warning("Using individual model files (create deployment package for better performance)")
            return model, scaler, pca, feature_names, model_info
            
        else:
            st.error(" Model files not found. Please train the model first.")
            return None, None, None, None, None
            
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None


model, scaler, pca, expected_features, model_info = load_deployment_package()

st.title('❤️ Heart Disease Prediction App')

if model and scaler and expected_features:
    
    st.sidebar.success(f"Model loaded! Expecting {len(expected_features)} features")
    
    
    using_pca = any('PC' in str(feat) for feat in expected_features) if expected_features else False
    if using_pca:
        st.sidebar.warning(" Model uses PCA features")
    else:
        st.sidebar.info("Model uses original features")
    
    
    with st.sidebar.expander("Model Information"):
        st.write(f"**Model Type:** {model_info.get('model_type', 'Unknown')}")
        st.write(f"**Version:** {model_info.get('version', '1.0')}")
        st.write(f"**Accuracy:** {model_info.get('best_accuracy', 'Unknown')}")
        st.write(f"**Features:** {len(expected_features)}")
    
    st.sidebar.header('Patient Information')
    input_data = {}
    
    
    st.sidebar.subheader('Demographic Information')
    if 'age' in expected_features or not using_pca:
        input_data['age'] = st.sidebar.slider('Age', 20, 100, 50)
    if 'sex' in expected_features or not using_pca:
        sex = st.sidebar.selectbox('Sex', ['Female', 'Male'])
        input_data['sex'] = 1 if sex == 'Male' else 0
    
    
    st.sidebar.subheader('Medical History')
    if 'cp' in expected_features or not using_pca:
        cp = st.sidebar.selectbox('Chest Pain Type', 
                                ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
        input_data['cp'] = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(cp) + 1
    if 'trestbps' in expected_features or not using_pca:
        input_data['trestbps'] = st.sidebar.slider('Resting BP (mm Hg)', 90, 200, 120)
    if 'chol' in expected_features or not using_pca:
        input_data['chol'] = st.sidebar.slider('Cholesterol (mg/dl)', 100, 600, 200)
    if 'fbs' in expected_features or not using_pca:
        fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
        input_data['fbs'] = 1 if fbs == 'Yes' else 0
    if 'restecg' in expected_features or not using_pca:
        restecg = st.sidebar.selectbox('Resting ECG', 
                                     ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
        input_data['restecg'] = ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'].index(restecg)
    
    
    st.sidebar.subheader('Exercise Metrics')
    if 'thalach' in expected_features or not using_pca:
        input_data['thalach'] = st.sidebar.slider('Max Heart Rate', 70, 220, 150)
    if 'exang' in expected_features or not using_pca:
        exang = st.sidebar.selectbox('Exercise Induced Angina', ['No', 'Yes'])
        input_data['exang'] = 1 if exang == 'Yes' else 0
    if 'oldpeak' in expected_features or not using_pca:
        input_data['oldpeak'] = st.sidebar.slider('ST Depression', 0.0, 6.0, 1.0, 0.1)
    if 'slope' in expected_features or not using_pca:
        slope = st.sidebar.selectbox('ST Slope', ['Upsloping', 'Flat', 'Downsloping'])
        input_data['slope'] = ['Upsloping', 'Flat', 'Downsloping'].index(slope) + 1
    if 'ca' in expected_features or not using_pca:
        input_data['ca'] = st.sidebar.slider('Major Vessels', 0, 3, 1)
    if 'thal' in expected_features or not using_pca:
        thal = st.sidebar.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])
        input_data['thal'] = ['Normal', 'Fixed Defect', 'Reversible Defect'].index(thal) + 3
    
    
    input_df = pd.DataFrame([input_data])
    
    
    for feature in expected_features:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    
    input_df = input_df[expected_features]
    
    
    st.subheader('Patient Data Summary')
    st.dataframe(input_df)
    
    
    if st.button('Predict Heart Disease Risk', type='primary'):
        try:
            with st.spinner('Analyzing health data...'):
                
                input_scaled = scaler.transform(input_df)
                
                
                if pca is not None and using_pca:
                    input_transformed = pca.transform(input_scaled)
                    st.info(f" Applied PCA transformation")
                else:
                    input_transformed = input_scaled
                
                # Make prediction
                prediction = model.predict(input_transformed)
                probability = model.predict_proba(input_transformed)
                
                # Display results
                st.subheader('📊 Prediction Result')
                
                if prediction[0] == 1:
                    st.error('🚨 **High Risk: Heart disease likely**')
                    st.write("Recommendation: Please consult a healthcare professional for further evaluation and testing.")
                else:
                    st.success('✅ **Low Risk: Heart disease unlikely**')
                    st.write("Recommendation: Maintain a healthy lifestyle with regular check-ups.")
                
                # Probability metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric('Probability of No Disease', f'{probability[0][0]:.2%}')
                with col2:
                    st.metric('Probability of Disease', f'{probability[0][1]:.2%}')
                
                # Risk visualization
                st.subheader('📈 Risk Visualization')
                risk_data = pd.DataFrame({
                    'Risk Level': ['No Disease', 'Disease'],
                    'Probability': [probability[0][0], probability[0][1]]
                })
                st.bar_chart(risk_data.set_index('Risk Level'))
                
                # Risk interpretation
                st.subheader('💡 Risk Interpretation')
                risk_score = probability[0][1]
                if risk_score < 0.2:
                    st.info("**Low Risk** (0-20%): Very unlikely to have heart disease")
                elif risk_score < 0.4:
                    st.info("**Mild Risk** (20-40%): Low probability of heart disease")
                elif risk_score < 0.6:
                    st.warning("**Moderate Risk** (40-60%): Uncertain - further evaluation recommended")
                elif risk_score < 0.8:
                    st.warning("**High Risk** (60-80%): Likely to have heart disease")
                else:
                    st.error("**Very High Risk** (80-100%): Very high likelihood of heart disease")
            
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            st.write("This might be due to:")
            st.write("- Feature mismatch between input and model expectations")
            st.write("- Missing or incorrect scaling")
            st.write("- Please ensure all model files are properly trained")

else:
    st.warning("⚠️ Please train the model first by running the complete pipeline:")
    st.write("1. **Data Preprocessing** (`01_data_preprocessing.py`)")
    st.write("2. **PCA Analysis** (`02_pca_analysis.py`)")
    st.write("3. **Feature Selection** (`03_feature_selection.py`)")
    st.write("4. **Model Training** (`04_supervised_learning.py`)")
    st.write("5. **Hyperparameter Tuning** (`06_hyperparameter_tuning.py`)")
    
    st.info("💡 After training, the model will be ready for predictions!")

# Add footer with information
st.sidebar.header('About')
st.sidebar.info("""
**Heart Disease Prediction App**

This app uses machine learning to predict heart disease risk based on clinical measurements.

**Features used:**
- Demographic information (Age, Sex)
- Medical history (BP, Cholesterol, ECG)
- Exercise-induced metrics
- Clinical measurements

**Disclaimer:** 
This tool is for educational purposes only. Always consult healthcare professionals for medical advice.
""")

# Debug information
with st.sidebar.expander("🔍 Debug Information"):
    if model is not None:
        try:
            st.write("**Model expects features:**", expected_features)
            st.write("**Number of features:**", len(expected_features))
            st.write("**Using PCA:**", using_pca)
        except:
            st.write("Could not access model feature information")
    
    st.write("**Files found:**")
    st.write(f"- Model: {os.path.exists('C:/studies/sprints/Heart_Disease_Project/models/final_model.pkl')}")
    st.write(f"- Scaler: {os.path.exists('C:/studies/sprints/Heart_Disease_Project/models/scaler.pkl')}")
    st.write(f"- PCA: {os.path.exists('C:/studies/sprints/Heart_Disease_Project/models/pca_model.pkl')}")
    st.write(f"- Deployment: {os.path.exists('C:/studies/sprints/Heart_Disease_Project/deployment/model_package.pkl')}")