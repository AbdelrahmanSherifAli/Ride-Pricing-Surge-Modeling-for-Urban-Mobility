import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, sin, cos, sqrt, atan2

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix
from xgboost import XGBRegressor, XGBClassifier

# Page configuration
st.set_page_config(
    page_title="Ride Price & Surge Prediction",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the data"""
    try:
        cab_data = pd.read_csv('cab_rides.csv')
        weather_data = pd.read_csv('weather.csv')
        return cab_data, weather_data
    except FileNotFoundError:
        st.error("Data files not found. Please ensure 'cab_rides.csv' and 'weather.csv' are in the same directory.")
        return None, None

def haversine(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between two points"""
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

@st.cache_data
def preprocess_data(cab_data, weather_data):
    """Preprocess the data exactly as in the notebook"""
    # Location coordinates
    location_coords = {
        'Haymarket Square': (42.3638, -71.0589),
        'Back Bay': (42.3503, -71.0810),
        'North Station': (42.3656, -71.0616),
        'South Station': (42.3522, -71.0552),
        'Fenway': (42.3467, -71.0972),
        'Financial District': (42.3555, -71.0565),
        'Beacon Hill': (42.3588, -71.0707),
        'West End': (42.3645, -71.0631)
    }
    
    # Clean cab data
    cab_clean = cab_data.copy()
    cab_clean = cab_clean.drop(['id', 'product_id'], axis=1, errors='ignore')
    cab_clean = cab_clean.dropna(subset=['price'])
    cab_clean['datetime'] = pd.to_datetime(cab_clean['time_stamp'], unit='ms')
    
    # Clean weather data
    weather_clean = weather_data.copy()
    weather_clean['rain'] = weather_clean['rain'].fillna(0)
    weather_clean['datetime'] = pd.to_datetime(weather_clean['time_stamp'], unit='s')
    
    # Time alignment - aggregate to hourly
    cab_clean['hour'] = cab_clean['datetime'].dt.floor('H')
    weather_clean['hour'] = weather_clean['datetime'].dt.floor('H')
    
    cab_hourly = cab_clean.groupby('hour').agg(
        avg_price=('price', 'mean'),
        avg_distance=('distance', 'mean'),
        avg_surge=('surge_multiplier', 'mean'),
        ride_count=('price', 'count')
    ).reset_index()
    
    weather_hourly = weather_clean.groupby('hour').agg(
        avg_temp=('temp', 'mean'),
        avg_rain=('rain', 'mean'),
        avg_clouds=('clouds', 'mean'),
        avg_humidity=('humidity', 'mean'),
        avg_wind=('wind', 'mean'),
        avg_pressure=('pressure', 'mean')
    ).reset_index()
    
    # Merge
    cab_weather = cab_clean.merge(weather_hourly, on='hour', how='left')
    
    # Feature engineering - geospatial
    cab_weather['pickup_lat'] = cab_weather['source'].map(lambda x: location_coords.get(x, (np.nan, np.nan))[0])
    cab_weather['pickup_lon'] = cab_weather['source'].map(lambda x: location_coords.get(x, (np.nan, np.nan))[1])
    cab_weather['dropoff_lat'] = cab_weather['destination'].map(lambda x: location_coords.get(x, (np.nan, np.nan))[0])
    cab_weather['dropoff_lon'] = cab_weather['destination'].map(lambda x: location_coords.get(x, (np.nan, np.nan))[1])
    
    # Temporal features (exactly as notebook)
    cab_weather['time_stamp'] = pd.to_datetime(cab_weather['time_stamp'], unit='ms')
    cab_weather['year'] = cab_weather['time_stamp'].dt.year
    cab_weather['month'] = cab_weather['time_stamp'].dt.month
    cab_weather['day'] = cab_weather['time_stamp'].dt.day
    cab_weather['day_of_week'] = cab_weather['time_stamp'].dt.dayofweek
    cab_weather['hour'] = cab_weather['time_stamp'].dt.hour
    cab_weather['is_rushhour'] = cab_weather['hour'].isin([7,8,9,16,17,18,19]).astype(int)
    cab_weather['is_weekend'] = cab_weather['day_of_week'].isin([5,6]).astype(int)
    
    # Frequency encoding
    source_freq = cab_weather['source'].value_counts(normalize=True)
    cab_weather['source_freq'] = cab_weather['source'].map(source_freq)
    dest_freq = cab_weather['destination'].value_counts(normalize=True)
    cab_weather['destination_freq'] = cab_weather['destination'].map(dest_freq)
    
    # One-hot encode cab_type and name
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    encoded = ohe.fit_transform(cab_weather[['cab_type', 'name']])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(['cab_type', 'name']))
    cab_weather = pd.concat([cab_weather.drop(columns=['cab_type', 'name', 'source', 'destination']), encoded_df], axis=1)
    
    return cab_weather, ohe

def train_models(cab_weather):
    """Train the models exactly as in the notebook"""
    # Drop columns as per notebook
    drop_cols = ['geo_distance_km', 'pickup_lat', 'pickup_lon', 'dropoff_lat', 'dropoff_lon', 'datetime', 'time_stamp']
    cab_weather_clean = cab_weather.drop(columns=drop_cols, errors='ignore')
    
    # Prepare data for regression
    X_reg = cab_weather_clean.drop(columns=['price', 'surge_multiplier'], errors='ignore')
    y_reg = cab_weather_clean['price']
    
    # Split data (70% train, 15% val, 15% test)
    X_temp, X_test_reg, y_temp, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
    X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Preprocessing for regression
    num_cols_reg = X_train_reg.select_dtypes(include=['int64', 'float64']).columns
    imputer_reg = SimpleImputer(strategy='mean')
    scaler_reg = StandardScaler()
    
    X_train_reg[num_cols_reg] = imputer_reg.fit_transform(X_train_reg[num_cols_reg])
    X_train_reg[num_cols_reg] = scaler_reg.fit_transform(X_train_reg[num_cols_reg])
    X_val_reg[num_cols_reg] = imputer_reg.transform(X_val_reg[num_cols_reg])
    X_val_reg[num_cols_reg] = scaler_reg.transform(X_val_reg[num_cols_reg])
    X_test_reg[num_cols_reg] = imputer_reg.transform(X_test_reg[num_cols_reg])
    X_test_reg[num_cols_reg] = scaler_reg.transform(X_test_reg[num_cols_reg])
    
    # Train regression model
    reg_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
    )
    reg_model.fit(X_train_reg, y_train_reg)
    
    # Prepare data for classification
    cab_weather_cls = cab_weather_clean.copy()
    cab_weather_cls['is_surge'] = (cab_weather_cls['surge_multiplier'] > 1).astype(int)
    
    X_cls = cab_weather_cls.drop(columns=['price', 'surge_multiplier', 'is_surge'], errors='ignore')
    y_cls = cab_weather_cls['is_surge']
    
    # Split data (70% train, 15% val, 15% test)
    X_temp_cls, X_test_cls, y_temp_cls, y_test_cls = train_test_split(
        X_cls, y_cls, test_size=0.3, random_state=42, stratify=y_cls
    )
    X_train_cls, X_val_cls, y_train_cls, y_val_cls = train_test_split(
        X_temp_cls, y_temp_cls, test_size=0.5, random_state=42, stratify=y_temp_cls
    )
    
    # Preprocessing for classification
    num_cols_cls = X_train_cls.select_dtypes(include=['int64', 'float64']).columns
    imputer_cls = SimpleImputer(strategy='median')
    scaler_cls = StandardScaler()
    
    X_train_cls[num_cols_cls] = scaler_cls.fit_transform(imputer_cls.fit_transform(X_train_cls[num_cols_cls]))
    X_val_cls[num_cols_cls] = scaler_cls.transform(imputer_cls.transform(X_val_cls[num_cols_cls]))
    X_test_cls[num_cols_cls] = scaler_cls.transform(imputer_cls.transform(X_test_cls[num_cols_cls]))
    
    # Calculate scale_pos_weight
    neg = (y_train_cls == 0).sum()
    pos = (y_train_cls == 1).sum()
    scale_pos_weight = neg / pos
    
    # Train classification model
    cls_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    cls_model.fit(X_train_cls, y_train_cls)
    
    return {
        'reg_model': reg_model,
        'cls_model': cls_model,
        'imputer_reg': imputer_reg,
        'scaler_reg': scaler_reg,
        'imputer_cls': imputer_cls,
        'scaler_cls': scaler_cls,
        'X_test_reg': X_test_reg,
        'y_test_reg': y_test_reg,
        'X_test_cls': X_test_cls,
        'y_test_cls': y_test_cls,
        'feature_names': X_reg.columns.tolist(),
        'num_cols_reg': num_cols_reg.tolist(),
        'num_cols_cls': num_cols_cls.tolist()
    }

def main():
    st.markdown('<h1 class="main-header"> Ride Price & Surge Prediction</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ["Price Prediction", "Surge Prediction", "Model Performance", "Data Exploration"])
    
    # Load data
    cab_data, weather_data = load_data()
    
    if cab_data is None or weather_data is None:
        st.stop()
    
    # Preprocess data
    with st.spinner("Preprocessing data..."):
        cab_weather, ohe = preprocess_data(cab_data, weather_data)
    
    # Train or load models
    if 'models' not in st.session_state:
        with st.spinner("Training models (this may take a few minutes)..."):
            st.session_state.models = train_models(cab_weather)
    
    models = st.session_state.models
    
    # Page routing
    if page == "Price Prediction":
        show_price_prediction(models, ohe, cab_data, cab_weather)
    elif page == "Surge Prediction":
        show_surge_prediction(models, ohe, cab_data, cab_weather)
    elif page == "Model Performance":
        show_model_performance(models)
    elif page == "Data Exploration":
        show_data_exploration(cab_data, weather_data, cab_weather)

def show_price_prediction(models, ohe, cab_data, cab_weather):
    st.header(" Ride Price Prediction")
    st.markdown("Predict the estimated price for a cab ride based on various features.")
    
    # Initialize session state for presets
    if 'price_preset' not in st.session_state:
        st.session_state.price_preset = 'default'
    
    # Quick start guide
    # st.success("""
    # ** Quick Start (بداية سريعة):** 
    # 1. Click **" Default"** button to fill all fields automatically
    # 2. Or use **" Rush Hour"**, **" Weekend"**, or **" Rainy Day"** presets
    # 3. Click **"Predict Price"** button - that's it!
    
    # ** IMPORTANT:** Data is from **Nov-Dec 2018**. The date is automatically set correctly!
    # """)
    
    # Valid ranges guide
    with st.expander(" Valid Input Ranges (To Avoid Errors)"):
        st.markdown("""
        **Ride Details:**
        - **Distance**: 0.0 - 10.0 miles (typical: 2-5 miles)
        
        **Time:**
        - **Date**: Must be between Nov 1, 2018 - Dec 31, 2018 (automatically set!)
        - **Hour**: 0-23 (0 = midnight, 12 = noon, 17 = 5 PM)
        - **Minute**: 0-59 (in 15-minute intervals)
        
        **Weather Conditions:**
        - **Temperature**: -20°F to 120°F (typical: 20-80°F)
        - **Rain**: 0.0 - 50.0 mm (0 = no rain, >0 = rainy)
        - **Cloud Coverage**: 0.0 - 1.0 (0 = clear, 1 = overcast)
        - **Humidity**: 0.0 - 1.0 (typical: 0.4-0.8)
        - **Wind Speed**: 0.0 - 30.0 m/s (typical: 0-15 m/s)
        - **Pressure**: 950.0 - 1050.0 hPa (normal: 980-1040 hPa)
        
        **💡 Tips:**
        - Use the **preset buttons** (Default, Rush Hour, Weekend, Rainy Day) for easy setup
        - All fields have default values - you can use them as starting points
        - Rain is the most important weather factor for price prediction
        - Rush hours (7-9 AM, 4-7 PM) typically have higher prices
        - Weekends usually have higher prices than weekdays
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ride Details")
        
        cab_type = st.selectbox(
            "Cab Type",
            ["Uber", "Lyft"],
            help="Choose between Uber or Lyft. Uber typically has more service options."
        )
        
        name = st.selectbox(
            "Service Type",
            ["UberX", "UberPool", "UberXL", "UberBlack", "UberSUV", "UberLux",
             "Lyft", "Lyft XL", "Lux", "Lux Black", "Lux Black XL", "Shared"],
            help="""Service tiers (cheapest to most expensive):
            - Shared/Pool: Cheapest, shared ride
            - UberX/Lyft: Standard ride
            - UberXL/Lyft XL: Larger vehicle
            - Black/SUV: Premium vehicles
            - Lux: Luxury vehicles (most expensive)"""
        )
        
        source = st.selectbox(
            "Pickup Location",
            ["Haymarket Square", "Back Bay", "North Station", "South Station",
             "Fenway", "Financial District", "Beacon Hill", "West End"],
            help="Select where you'll be picked up. Popular locations may have surge pricing."
        )
        
        destination = st.selectbox(
            "Dropoff Location",
            ["Haymarket Square", "Back Bay", "North Station", "South Station",
             "Fenway", "Financial District", "Beacon Hill", "West End"],
            help="Select your destination. Distance between pickup and dropoff affects price."
        )
        
        distance = st.slider(
            "Distance (miles)",
            0.0, 10.0, 2.5, 0.1,
            help="Distance of the ride in miles. Longer rides cost more. Typical city rides are 2-5 miles."
        )
    
    with col2:
        st.subheader("Time & Weather")
        
        # Quick Fill Presets
        st.markdown("** Quick Fill (اختيار سريع):**")
        preset_cols = st.columns(4)
        with preset_cols[0]:
            if st.button(" Rush Hour", key="preset_rush"):
                st.session_state.price_preset = 'rush'
        with preset_cols[1]:
            if st.button(" Weekend", key="preset_weekend"):
                st.session_state.price_preset = 'weekend'
        with preset_cols[2]:
            if st.button(" Rainy Day", key="preset_rainy"):
                st.session_state.price_preset = 'rainy'
        with preset_cols[3]:
            if st.button(" Default", key="preset_default"):
                st.session_state.price_preset = 'default'
        
        # Show current preset
        preset_names = {
            'rush': ' Rush Hour',
            'weekend': 'Weekend',
            'rainy': ' Rainy Day',
            'default': ' Default'
        }
        if st.session_state.price_preset in preset_names:
            st.info(f"Current preset: **{preset_names[st.session_state.price_preset]}**")
        
        # Set default date to December 2018 (data range)
        default_date = datetime(2018, 12, 15).date()
        
        ride_date = st.date_input(
            "Ride Date",
            value=default_date,
            min_value=datetime(2018, 11, 1).date(),
            max_value=datetime(2018, 12, 31).date(),
            help=" IMPORTANT: Data is from Nov-Dec 2018. Use dates in this range for accurate predictions!"
        )
        
        # Use sliders for easier time selection
        col_hour, col_min = st.columns(2)
        with col_hour:
            # Default hour based on preset
            if st.session_state.price_preset == 'rush':
                default_hour = 8
            else:
                default_hour = 14
            ride_hour = st.slider(
                "Hour (24-hour format)",
                0, 23, default_hour,
                help="Select the hour: 0-23 (0 = midnight, 12 = noon, 17 = 5 PM). Rush hours: 7-9 AM, 4-7 PM"
            )
        with col_min:
            ride_minute = st.slider(
                "Minute",
                0, 59, 30,
                step=15,
                help="Select the minute (in 15-minute intervals)"
            )
        
        # Create datetime object
        from datetime import time as dt_time
        ride_time = dt_time(hour=ride_hour, minute=ride_minute, second=0)
        ride_datetime = datetime.combine(ride_date, ride_time)
        
        # Display the selected time
        st.info(f" Selected Time: {ride_datetime.strftime('%A, %B %d, %Y at %I:%M %p')}")
    
        # Show rush hour indicator
        is_rush = ride_datetime.hour in [7,8,9,16,17,18,19]
        is_weekend = ride_datetime.weekday() >= 5
        if is_rush:
            st.info(" Rush Hour - Prices may be higher")
        if is_weekend:
            st.info(" Weekend - Prices may be higher")
        
        st.markdown("---")
        st.markdown("**Weather Conditions** (affect demand and pricing)")
        
        # Set default values based on presets
        if st.session_state.price_preset == 'rainy':
            default_temp = 45.0
            default_rain = 5.0
            default_clouds = 0.8
            default_humidity = 0.75
            default_wind = 8.0
        elif st.session_state.price_preset == 'rush':
            default_temp = 50.0
            default_rain = 0.0
            default_clouds = 0.4
            default_humidity = 0.6
            default_wind = 5.0
        elif st.session_state.price_preset == 'weekend':
            default_temp = 55.0
            default_rain = 0.0
            default_clouds = 0.5
            default_humidity = 0.65
            default_wind = 6.0
        else:  # default
            default_temp = 50.0
            default_rain = 0.0
            default_clouds = 0.5
            default_humidity = 0.6
            default_wind = 5.0
        
        avg_temp = st.number_input(
            "Temperature (°F)",
            min_value=-20.0,
            max_value=120.0,
            value=default_temp,
            step=1.0,
            help="Air temperature in Fahrenheit. Typical range: 20-80°F. Extreme temperatures increase demand."
        )
        avg_rain = st.number_input(
            "Rain (mm)",
            min_value=0.0,
            max_value=50.0,
            value=default_rain,
            step=0.1,
            help="Rainfall in millimeters. 0 = no rain, 0.1-5 = light rain, 5-15 = moderate, >15 = heavy rain. Rain significantly increases prices!"
        )
        avg_clouds = st.slider(
            "Cloud Coverage",
            0.0, 1.0, default_clouds, 0.01,
            help="Cloud coverage from 0 (clear sky) to 1 (completely overcast). Typical: 0.3-0.7"
        )
        avg_humidity = st.slider(
            "Humidity",
            0.0, 1.0, default_humidity, 0.01,
            help="Humidity level from 0 (dry) to 1 (very humid). Typical range: 0.4-0.8. High humidity may increase demand."
        )
        avg_wind = st.number_input(
            "Wind Speed (m/s)",
            min_value=0.0,
            max_value=30.0,
            value=default_wind,
            step=0.5,
            help="Wind speed in meters per second. Typical: 0-15 m/s (0-5 = calm, 5-10 = moderate, >10 = strong wind)"
        )
        avg_pressure = st.number_input(
            "Pressure (hPa)",
            min_value=950.0,
            max_value=1050.0,
            value=1013.0,
            step=1.0,
            help="Atmospheric pressure in hectopascals. Normal range: 980-1040 hPa. Standard sea level: ~1013 hPa"
        )
    
    # Prepare input
    if st.button("Predict Price", type="primary"):
        try:
            # Calculate frequency maps from original data
            source_freq_map = cab_data['source'].value_counts(normalize=True).to_dict()
            dest_freq_map = cab_data['destination'].value_counts(normalize=True).to_dict()
            
            source_freq = source_freq_map.get(source, 0.125)
            dest_freq = dest_freq_map.get(destination, 0.125)
            
            # Create input data matching the training structure EXACTLY
            input_data = pd.DataFrame({
                'cab_type': [cab_type],
                'name': [name],
                'source': [source],
                'destination': [destination],
                'distance': [distance],
                'year': [int(ride_datetime.year)],
                'month': [int(ride_datetime.month)],
                'day': [int(ride_datetime.day)],
                'day_of_week': [int(ride_datetime.weekday())],
                'hour': [int(ride_datetime.hour)],
                'is_rushhour': [int(1 if ride_datetime.hour in [7,8,9,16,17,18,19] else 0)],
                'is_weekend': [int(1 if ride_datetime.weekday() >= 5 else 0)],
                'source_freq': [float(source_freq)],
                'destination_freq': [float(dest_freq)],
                'avg_temp': [float(avg_temp)],
                'avg_rain': [float(avg_rain)],
                'avg_clouds': [float(avg_clouds)],
                'avg_humidity': [float(avg_humidity)],
                'avg_wind': [float(avg_wind)],
                'avg_pressure': [float(avg_pressure)]
            })
            
            # Step 1: One-hot encode cab_type and name
            encoded = ohe.transform(input_data[['cab_type', 'name']])
            encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(['cab_type', 'name']))
            
            # Step 2: Drop categorical columns and combine with encoded
            features = pd.concat([
                input_data.drop(columns=['cab_type', 'name', 'source', 'destination']),
                encoded_df
            ], axis=1)
            
            # Step 3: Ensure columns match training features exactly
            feature_cols = models['feature_names']
            
            # Create a dataframe with all training features, initialized to 0
            features_aligned = pd.DataFrame(0.0, index=[0], columns=feature_cols, dtype=float)
            
            # Fill in the values we have
            for col in features.columns:
                if col in feature_cols:
                    val = features[col].iloc[0] if hasattr(features[col], 'iloc') else features[col].values[0]
                    features_aligned[col] = val
            
            # Step 4: Apply imputation and scaling
            num_cols = models['num_cols_reg']
            
            # Ensure all required numerical columns exist
            for col in num_cols:
                if col not in features_aligned.columns:
                    features_aligned[col] = 0.0
            
            # Reorder to match training order exactly
            features_for_imputer = features_aligned[num_cols].copy()
            
            # Transform with imputer
            features_imputed = pd.DataFrame(
                models['imputer_reg'].transform(features_for_imputer),
                columns=num_cols,
                index=features_aligned.index
            )
            
            # Transform with scaler
            features_scaled = pd.DataFrame(
                models['scaler_reg'].transform(features_imputed),
                columns=num_cols,
                index=features_aligned.index
            )
            
            # Update the aligned features with scaled values
            features_aligned[num_cols] = features_scaled
            
            # Ensure exact column order as training
            features_aligned = features_aligned[feature_cols]
            
            # Predict
            prediction = models['reg_model'].predict(features_aligned)[0]
        
            st.success(f"**Predicted Price: ${prediction:.2f}**")
            
            # Show feature importance
            st.subheader("Top 5 Most Important Features")
            feature_importance = pd.Series(
                models['reg_model'].feature_importances_,
                index=feature_cols
            ).sort_values(ascending=False).head(5)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=feature_importance.values, y=feature_importance.index, ax=ax)
            ax.set_xlabel('Importance Score')
            ax.set_ylabel('Features')
            ax.set_title('Top 5 Feature Importance')
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.warning("""
            **Common Issues & Solutions:**
            1. **Feature mismatch error**: Make sure all inputs are within valid ranges (see guide above)
            2. **Missing values**: All fields must have values - check that no field is empty
            3. **Invalid ranges**: Use the sliders and number inputs with their min/max limits
            4. **Date/Time**: Make sure date and time are selected correctly
            
            **Try:**
            - Refresh the page and try again
            - Use default values for weather if unsure
            - Make sure date is not too far in the past or future
            """)
            # Show detailed error for debugging
            with st.expander(" Technical Details (for debugging)"):
                st.code(str(e))

def show_surge_prediction(models, ohe, cab_data, cab_weather):
    st.header(" Surge Prediction")
    st.markdown("Predict whether surge pricing will be active for a ride.")
    
    # Initialize session state for presets
    if 'surge_preset' not in st.session_state:
        st.session_state.surge_preset = 'default'
    
    # Quick start guide
    st.success("""
    ** Quick Start (بداية سريعة):**
    1. Click **" Default"** button to fill all fields automatically
    2. Or use **" Rush Hour"**, **" Weekend"**, or **" Rainy Day"** presets
    3. Click **"Predict Surge"** button - that's it!
    
    **IMPORTANT:** Data is from **Nov-Dec 2018**. The date is automatically set correctly!
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ride Details")
        
        cab_type = st.selectbox(
            "Cab Type",
            ["Uber", "Lyft"],
            key="surge_cab",
            help="Choose between Uber or Lyft. Both can have surge pricing."
        )
        
        name = st.selectbox(
            "Service Type",
            ["UberX", "UberPool", "UberXL", "UberBlack", "UberSUV", "UberLux",
             "Lyft", "Lyft XL", "Lux", "Lux Black", "Lux Black XL", "Shared"],
            key="surge_name",
            help="Premium services may have different surge patterns than standard services."
        )
        
        source = st.selectbox(
            "Pickup Location",
            ["Haymarket Square", "Back Bay", "North Station", "South Station",
             "Fenway", "Financial District", "Beacon Hill", "West End"],
            key="surge_source",
            help="Popular locations (Financial District, Back Bay) are more likely to have surge pricing."
        )
        
        destination = st.selectbox(
            "Dropoff Location",
            ["Haymarket Square", "Back Bay", "North Station", "South Station",
             "Fenway", "Financial District", "Beacon Hill", "West End"],
            key="surge_dest",
            help="Destination location affects surge probability."
        )
        
        distance = st.slider(
            "Distance (miles)",
            0.0, 10.0, 2.5, 0.1,
            key="surge_dist",
            help="Distance doesn't directly cause surge, but longer rides during high demand periods may have surge."
        )
    
    with col2:
        st.subheader("Time & Weather")
        
        # Quick Fill Presets for Surge Prediction
        st.markdown("** Quick Fill (اختيار سريع):**")
        preset_cols_surge = st.columns(4)
        with preset_cols_surge[0]:
            if st.button(" Rush Hour", key="surge_preset_rush"):
                st.session_state.surge_preset = 'rush'
        with preset_cols_surge[1]:
            if st.button(" Weekend", key="surge_preset_weekend"):
                st.session_state.surge_preset = 'weekend'
        with preset_cols_surge[2]:
            if st.button(" Rainy Day", key="surge_preset_rainy"):
                st.session_state.surge_preset = 'rainy'
        with preset_cols_surge[3]:
            if st.button(" Default", key="surge_preset_default"):
                st.session_state.surge_preset = 'default'
        
        # Show current preset
        preset_names_surge = {
            'rush': ' Rush Hour',
            'weekend': ' Weekend',
            'rainy': ' Rainy Day',
            'default': '✨ Default'
        }
        if st.session_state.surge_preset in preset_names_surge:
            st.info(f"Current preset: **{preset_names_surge[st.session_state.surge_preset]}**")
        
        # Set default date to December 2018 (data range)
        default_date_surge = datetime(2018, 12, 15).date()
        
        ride_date = st.date_input(
            "Ride Date",
            value=default_date_surge,
            min_value=datetime(2018, 11, 1).date(),
            max_value=datetime(2018, 12, 31).date(),
            key="surge_date",
            help="⚠️ IMPORTANT: Data is from Nov-Dec 2018. Use dates in this range for accurate predictions!"
        )
        
        # Use sliders for easier time selection
        col_hour, col_min = st.columns(2)
        with col_hour:
            # Default hour based on preset
            if st.session_state.surge_preset == 'rush':
                default_hour_surge = 8
            else:
                default_hour_surge = 14
            ride_hour = st.slider(
                "Hour (24-hour format)",
                0, 23, default_hour_surge,
                key="surge_hour_slider",
                help="Select the hour: 0-23 (0 = midnight, 12 = noon, 17 = 5 PM). Rush hours: 7-9 AM, 4-7 PM"
            )
        with col_min:
            ride_minute = st.slider(
                "Minute",
                0, 59, 30,
                step=15,
                key="surge_min_slider",
                help="Select the minute (in 15-minute intervals)"
            )
        
        # Create datetime object
        from datetime import time as dt_time
        ride_time = dt_time(hour=ride_hour, minute=ride_minute, second=0)
        ride_datetime = datetime.combine(ride_date, ride_time)
        
        # Display the selected time
        st.info(f" Selected Time: {ride_datetime.strftime('%A, %B %d, %Y at %I:%M %p')}")
        
        # Show indicators
        is_rush = ride_datetime.hour in [7,8,9,16,17,18,19]
        is_weekend = ride_datetime.weekday() >= 5
        if is_rush:
            st.warning(" Rush Hour - High surge probability")
        if is_weekend:
            st.warning(" Weekend - Higher surge probability")
        
        st.markdown("---")
        st.markdown("**Weather Conditions** (major factor for surge)")
        
        # Set default values based on presets
        if st.session_state.surge_preset == 'rainy':
            default_temp_surge = 45.0
            default_rain_surge = 5.0
            default_clouds_surge = 0.8
            default_humidity_surge = 0.75
            default_wind_surge = 8.0
        elif st.session_state.surge_preset == 'rush':
            default_temp_surge = 50.0
            default_rain_surge = 0.0
            default_clouds_surge = 0.4
            default_humidity_surge = 0.6
            default_wind_surge = 5.0
        elif st.session_state.surge_preset == 'weekend':
            default_temp_surge = 55.0
            default_rain_surge = 0.0
            default_clouds_surge = 0.5
            default_humidity_surge = 0.65
            default_wind_surge = 6.0
        else:  # default
            default_temp_surge = 50.0
            default_rain_surge = 0.0
            default_clouds_surge = 0.5
            default_humidity_surge = 0.6
            default_wind_surge = 5.0
        
        avg_temp = st.number_input(
            "Temperature (°F)",
            min_value=-20.0,
            max_value=120.0,
            value=default_temp_surge,
            step=1.0,
            key="surge_temp",
            help="Extreme temperatures (very hot or cold) increase demand and surge probability. Typical: 20-80°F"
        )
        
        avg_rain = st.number_input(
            "Rain (mm)",
            min_value=0.0,
            max_value=50.0,
            value=default_rain_surge,
            step=0.1,
            key="surge_rain",
            help="Rain significantly increases demand! 0 = no rain, >0 mm = higher surge probability. Heavy rain (>5mm) = very high surge."
        )
        
        avg_clouds = st.slider(
            "Cloud Coverage",
            0.0, 1.0, default_clouds_surge, 0.01,
            key="surge_clouds",
            help="Cloud coverage from 0 (clear sky) to 1 (completely overcast). Typical: 0.3-0.7"
        )
        
        avg_humidity = st.slider(
            "Humidity",
            0.0, 1.0, default_humidity_surge, 0.01,
            key="surge_humidity",
            help="Humidity level from 0 (dry) to 1 (very humid). High humidity may increase demand. Typical: 0.4-0.8"
        )
        
        avg_wind = st.number_input(
            "Wind Speed (m/s)",
            min_value=0.0,
            max_value=30.0,
            value=default_wind_surge,
            step=0.5,
            key="surge_wind",
            help="Wind speed in meters per second. Typical: 0-15 m/s (0-5 = calm, 5-10 = moderate, >10 = strong wind)"
        )
        
        avg_pressure = st.number_input(
            "Pressure (hPa)",
            min_value=950.0,
            max_value=1050.0,
            value=1013.0,
            step=1.0,
            key="surge_pressure",
            help="Atmospheric pressure in hectopascals. Normal range: 980-1040 hPa. Standard sea level: ~1013 hPa"
        )
    
    threshold = st.slider(
        "Surge Prediction Threshold",
        0.0, 1.0, 0.3, 0.05,
        help="""Probability threshold for predicting surge:
        - 0.1-0.2: Very sensitive (predicts surge often, may have false positives)
        - 0.3-0.4: Balanced (recommended)
        - 0.5-0.7: Conservative (only predicts when very likely, may miss some surges)
        - 0.8+: Very conservative (rarely predicts surge)"""
    )
    
    if st.button("Predict Surge", type="primary"):
        try:
            # Calculate frequency maps from original data
            source_freq_map = cab_data['source'].value_counts(normalize=True).to_dict()
            dest_freq_map = cab_data['destination'].value_counts(normalize=True).to_dict()
            
            source_freq = source_freq_map.get(source, 0.125)
            dest_freq = dest_freq_map.get(destination, 0.125)
            
            # Create input data matching the training structure EXACTLY
            input_data = pd.DataFrame({
                'cab_type': [cab_type],
                'name': [name],
                'source': [source],
                'destination': [destination],
                'distance': [distance],
                'year': [int(ride_datetime.year)],
                'month': [int(ride_datetime.month)],
                'day': [int(ride_datetime.day)],
                'day_of_week': [int(ride_datetime.weekday())],
                'hour': [int(ride_datetime.hour)],
                'is_rushhour': [int(1 if ride_datetime.hour in [7,8,9,16,17,18,19] else 0)],
                'is_weekend': [int(1 if ride_datetime.weekday() >= 5 else 0)],
                'source_freq': [float(source_freq)],
                'destination_freq': [float(dest_freq)],
                'avg_temp': [float(avg_temp)],
                'avg_rain': [float(avg_rain)],
                'avg_clouds': [float(avg_clouds)],
                'avg_humidity': [float(avg_humidity)],
                'avg_wind': [float(avg_wind)],
                'avg_pressure': [float(avg_pressure)]
            })
            
            # Step 1: One-hot encode cab_type and name
            encoded = ohe.transform(input_data[['cab_type', 'name']])
            encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(['cab_type', 'name']))
            
            # Step 2: Drop categorical columns and combine with encoded
            features = pd.concat([
                input_data.drop(columns=['cab_type', 'name', 'source', 'destination']),
                encoded_df
            ], axis=1)
            
            # Step 3: Ensure columns match training features exactly
            feature_cols = models['feature_names']
            
            # Create a dataframe with all training features, initialized to 0
            features_aligned = pd.DataFrame(0.0, index=[0], columns=feature_cols, dtype=float)
            
            # Fill in the values we have
            for col in features.columns:
                if col in feature_cols:
                    val = features[col].iloc[0] if hasattr(features[col], 'iloc') else features[col].values[0]
                    features_aligned[col] = val
            
            # Step 4: Apply imputation and scaling
            num_cols = models['num_cols_cls']
            
            # Ensure all required numerical columns exist
            for col in num_cols:
                if col not in features_aligned.columns:
                    features_aligned[col] = 0.0
            
            # Reorder to match training order exactly
            features_for_imputer = features_aligned[num_cols].copy()
            
            # Transform with imputer
            features_imputed = pd.DataFrame(
                models['imputer_cls'].transform(features_for_imputer),
                columns=num_cols,
                index=features_aligned.index
            )
            
            # Transform with scaler
            features_scaled = pd.DataFrame(
                models['scaler_cls'].transform(features_imputed),
                columns=num_cols,
                index=features_aligned.index
            )
            
            # Update the aligned features with scaled values
            features_aligned[num_cols] = features_scaled
            
            # Ensure exact column order as training
            features_aligned = features_aligned[feature_cols]
            
            # Predict
            proba = models['cls_model'].predict_proba(features_aligned)[0][1]
            prediction = 1 if proba >= threshold else 0
            
            if prediction == 1:
                st.warning(f"** Surge Pricing Active!** (Probability: {proba:.2%})")
                st.info(" Tip: Consider waiting a few minutes or choosing a different pickup location to avoid surge pricing.")
            else:
                st.success(f"**✅ No Surge Pricing** (Probability: {proba:.2%})")
            
            # Show probability gauge
            st.progress(proba)
            
            # Show interpretation
            if proba < 0.2:
                st.info("Low surge probability - Good time to book!")
            elif proba < 0.5:
                st.info("Moderate surge probability - Surge may occur")
            else:
                st.warning("High surge probability - Surge likely!")
                
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.warning("""
            **Common Issues & Solutions:**
            1. **Feature mismatch error**: Make sure all inputs are within valid ranges
            2. **Missing values**: All fields must have values
            3. **Invalid ranges**: Use the sliders and number inputs with their min/max limits
            4. **Date/Time**: Make sure date and time are selected correctly
            
            **Try:**
            - Refresh the page and try again
            - Use default values for weather if unsure
            - Make sure date is not too far in the past or future
            """)
            # Show detailed error for debugging
            with st.expander("🔍 Technical Details (for debugging)"):
                st.code(str(e))

def show_model_performance(models):
    st.header("📊 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Regression Model (Price Prediction)")
        y_pred_reg = models['reg_model'].predict(models['X_test_reg'])
        
        mae = mean_absolute_error(models['y_test_reg'], y_pred_reg)
        rmse = np.sqrt(mean_squared_error(models['y_test_reg'], y_pred_reg))
        r2 = r2_score(models['y_test_reg'], y_pred_reg)
        
        st.metric("Mean Absolute Error (MAE)", f"${mae:.2f}")
        st.metric("Root Mean Squared Error (RMSE)", f"${rmse:.2f}")
        st.metric("R² Score", f"{r2:.4f}")
        
        # Prediction vs Actual plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(models['y_test_reg'], y_pred_reg, alpha=0.5)
        ax.plot([models['y_test_reg'].min(), models['y_test_reg'].max()],
                [models['y_test_reg'].min(), models['y_test_reg'].max()], 'r--', lw=2)
        ax.set_xlabel('Actual Price')
        ax.set_ylabel('Predicted Price')
        ax.set_title('Predicted vs Actual Price')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Classification Model (Surge Prediction)")
        y_pred_cls = models['cls_model'].predict(models['X_test_cls'])
        y_proba_cls = models['cls_model'].predict_proba(models['X_test_cls'])[:, 1]
        
        threshold = 0.3
        y_pred_thresh = (y_proba_cls >= threshold).astype(int)
        
        report = classification_report(models['y_test_cls'], y_pred_thresh, output_dict=True)
        
        st.metric("Accuracy", f"{report['accuracy']:.4f}")
        st.metric("Precision (Surge)", f"{report['1']['precision']:.4f}")
        st.metric("Recall (Surge)", f"{report['1']['recall']:.4f}")
        st.metric("F1-Score (Surge)", f"{report['1']['f1-score']:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(models['y_test_cls'], y_pred_thresh)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

def show_data_exploration(cab_data, weather_data, cab_weather):
    st.header("🔍 Data Exploration")
    
    tab1, tab2, tab3 = st.tabs(["Cab Data", "Weather Data", "Combined Analysis"])
    
    with tab1:
        st.subheader("Cab Ride Data Overview")
        st.dataframe(cab_data.head(10))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rides", len(cab_data))
            st.metric("Missing Prices", cab_data['price'].isnull().sum())
        with col2:
            st.metric("Average Price", f"${cab_data['price'].mean():.2f}")
            st.metric("Average Distance", f"{cab_data['distance'].mean():.2f} miles")
        
        # Price distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        cab_data['price'].hist(bins=50, ax=ax)
        ax.set_xlabel('Price ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('Price Distribution')
        st.pyplot(fig)
        
        # Price by service type
        fig, ax = plt.subplots(figsize=(12, 6))
        cab_data.groupby('name')['price'].mean().sort_values().plot(kind='barh', ax=ax)
        ax.set_xlabel('Average Price ($)')
        ax.set_ylabel('Service Type')
        ax.set_title('Average Price by Service Type')
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Weather Data Overview")
        st.dataframe(weather_data.head(10))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", len(weather_data))
            st.metric("Average Temperature", f"{weather_data['temp'].mean():.1f}°F")
        with col2:
            st.metric("Locations", weather_data['location'].nunique())
            st.metric("Rainy Days", (weather_data['rain'] > 0).sum())
        
        # Temperature distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        weather_data['temp'].hist(bins=30, ax=ax)
        ax.set_xlabel('Temperature (°F)')
        ax.set_ylabel('Frequency')
        ax.set_title('Temperature Distribution')
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Combined Analysis")
        
        # Price vs Distance
        fig, ax = plt.subplots(figsize=(10, 6))
        sample = cab_weather.sample(min(10000, len(cab_weather)))
        ax.scatter(sample['distance'], sample['price'], alpha=0.5)
        ax.set_xlabel('Distance (miles)')
        ax.set_ylabel('Price ($)')
        ax.set_title('Price vs Distance')
        st.pyplot(fig)
        
        # Price by rush hour
        if 'is_rushhour' in cab_weather.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            cab_weather.groupby('is_rushhour')['price'].mean().plot(kind='bar', ax=ax)
            ax.set_xlabel('Rush Hour (0=No, 1=Yes)')
            ax.set_ylabel('Average Price ($)')
            ax.set_title('Average Price: Rush Hour vs Normal Hours')
            ax.set_xticklabels(['Normal', 'Rush Hour'], rotation=0)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
