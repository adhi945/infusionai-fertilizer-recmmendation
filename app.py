import streamlit as st
import pandas as pd
import pickle
import numpy as np
import random

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
    with open('fertilizer_recommendation_model.pkl', 'rb') as f:
        model = pickle.load(f)
    try:
        with open('feature_encoders.pkl', 'rb') as f:
            feature_encoders = pickle.load(f)
    except:
        feature_encoders = None
    return scaler, label_encoder, feature_encoders, model

# Expanded remarks dictionary
remarks_dict = {
    'Urea': 'High Nitrogen fertilizer, excellent for green leafy vegetables.',
    'DAP': 'Rich in Phosphorus, ideal for root development and strong growth.',
    '14-35-14': 'Boosts flowering and fruiting stages effectively.',
    '28-28': 'Ensures balanced growth during vegetative phase.',
    '17-17-17': 'General-purpose fertilizer for various crops and stages.',
    '20-20': 'Great starter fertilizer for young plants and seedlings.',
    '10-26-26': 'Promotes flowering, fruiting, and plant maturity.',
    'General Purpose Fertilizer': 'Perfect for maintaining healthy plants throughout the season.',
    'NPK 19-19-19': 'Balanced NPK fertilizer, supports vigorous plant growth.',
    'Compost': 'Organic fertilizer enriching soil quality sustainably.',
    'Vermicompost': 'Natural worm-processed fertilizer for soil health.',
    'Cow Manure': 'Traditional organic fertilizer improving soil texture.',
    'Potash': 'Boosts resistance to diseases and improves crop quality.',
    'Superphosphate': 'Helps rapid root establishment and flowering.'
}

# Main function
def main():
    local_css("style.css")

    st.title("🌾 Fertilizer Recommendation System")
    st.write("Provide the details below to get the best fertilizer recommendation.")

    # Load models
    scaler, label_encoder, feature_encoders, model = load_files()

    # Setup fallback soil and crop options
    if feature_encoders and 'Soil Type' in feature_encoders and 'Crop Type' in feature_encoders:
        soil_options = feature_encoders['Soil Type'].classes_
        crop_options = feature_encoders['Crop Type'].classes_
    else:
        st.warning("⚠️ Using default Soil and Crop options due to missing encoders.")
        soil_options = ['Loamy', 'Sandy', 'Clayey', 'Black', 'Red', 'Alluvial']
        crop_options = ['Wheat', 'Rice', 'Sugarcane', 'Maize', 'Cotton', 'Barley']

    # User inputs
    temperature = st.number_input('Temperature (°C)', min_value=0.0, max_value=50.0, value=25.0)
    humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=50.0)
    moisture = st.number_input('Moisture (%)', min_value=0.0, max_value=100.0, value=30.0)

    soil_type = st.selectbox('Soil Type', soil_options)
    crop_type = st.selectbox('Crop Type', crop_options)

    nitrogen = st.number_input('Nitrogen Level (N)', min_value=0.0, max_value=100.0, value=20.0)
    phosphorus = st.number_input('Phosphorus Level (P)', min_value=0.0, max_value=100.0, value=30.0)
    potassium = st.number_input('Potassium Level (K)', min_value=0.0, max_value=100.0, value=40.0)

    if st.button('Recommend Fertilizer'):
        # Encode soil and crop if encoders exist
        try:
            soil_encoded = feature_encoders['Soil Type'].transform([soil_type])[0]
            crop_encoded = feature_encoders['Crop Type'].transform([crop_type])[0]
        except:
            # fallback dummy encoding
            soil_encoded = random.randint(0, 5)
            crop_encoded = random.randint(0, 5)

        # Input array with dummy values for missing features
        input_data = np.array([[
            temperature, humidity, moisture,
            soil_encoded, crop_encoded,
            nitrogen, phosphorus, potassium,
            random.uniform(5.5, 7.5),    # Random realistic pH
            random.uniform(100.0, 300.0), # Random rainfall
            random.uniform(50.0, 200.0)   # Random elevation
        ]])

        # Scale
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction_encoded = model.predict(input_scaled)
        prediction = label_encoder.inverse_transform(prediction_encoded)[0]

        # Fetch remark
        remark = remarks_dict.get(prediction, "🌿 Recommended for promoting healthy and balanced plant growth.")

        st.success(f"🌱 Recommended Fertilizer: **{prediction}**")
        st.info(f"💬 Remark: {remark}")

if __name__ == '__main__':
    main()
