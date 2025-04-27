import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Inject CSS from style.css
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load models and encoders
def load_files():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    with open('feature_encoders.pkl', 'rb') as f:
        feature_encoders = pickle.load(f)
    with open('fertilizer_recommendation_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return scaler, label_encoder, feature_encoders, model

# Remark mapping
remarks_dict = {
    'Urea': 'High Nitrogen fertilizer, good for leafy growth.',
    'DAP': 'High Phosphorus fertilizer, promotes root development.',
    '14-35-14': 'Balanced fertilizer for flowering and fruiting.',
    '28-28': 'Balanced fertilizer for overall growth.',
    '17-17-17': 'General-purpose fertilizer, suitable for most plants.',
    '20-20': 'Strong starter fertilizer, good for young plants.',
    '10-26-26': 'High Phosphorus and Potassium fertilizer for maturity.'
}

# Main function
def main():
    local_css("style.css")  # Apply custom styling

    st.title("🌾 Fertilizer Recommendation System")
    st.write("Provide the details below to get the best fertilizer recommendation.")

    # Load models and encoders
    scaler, label_encoder, feature_encoders, model = load_files()

    # Setup Soil and Crop Options
    try:
        soil_options = feature_encoders['Soil Type'].classes_
        crop_options = feature_encoders['Crop Type'].classes_
    except Exception as e:
        st.warning("⚠️ Using default Soil and Crop options due to missing encoders.")
        soil_options = ['Loamy', 'Sandy', 'Clayey', 'Black', 'Red', 'Alluvial']
        crop_options = ['Wheat', 'Rice', 'Sugarcane', 'Maize', 'Cotton', 'Barley']

    # User Inputs
    temperature = st.number_input('Temperature (°C)', min_value=0.0, max_value=50.0, value=25.0)
    humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=50.0)
    moisture = st.number_input('Moisture (%)', min_value=0.0, max_value=100.0, value=30.0)

    soil_type = st.selectbox('Soil Type', soil_options)
    crop_type = st.selectbox('Crop Type', crop_options)

    nitrogen = st.number_input('Nitrogen Level (N)', min_value=0.0, max_value=100.0, value=20.0)
    phosphorus = st.number_input('Phosphorus Level (P)', min_value=0.0, max_value=100.0, value=30.0)
    potassium = st.number_input('Potassium Level (K)', min_value=0.0, max_value=100.0, value=40.0)

    if st.button('Recommend Fertilizer'):
        # Encode categorical features
        try:
            soil_encoded = feature_encoders['Soil Type'].transform([soil_type])[0]
            crop_encoded = feature_encoders['Crop Type'].transform([crop_type])[0]
        except:
            soil_encoded = 0
            crop_encoded = 0

        # Create input array and add dummy values for missing 3 features
        input_data = np.array([[
            temperature, humidity, moisture,
            soil_encoded, crop_encoded,
            nitrogen, phosphorus, potassium,
            6.5,   # Dummy pH value
            200.0, # Dummy rainfall value
            100.0  # Dummy elevation value
        ]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction_encoded = model.predict(input_scaled)
        prediction = label_encoder.inverse_transform(prediction_encoded)[0]

        # Fetch remark
        remark = remarks_dict.get(prediction, "No specific remark available.")

        st.success(f"🌱 Recommended Fertilizer: **{prediction}**")
        st.info(f"💬 Remark: {remark}")

if __name__ == '__main__':
    main()
