# Ride Pricing & Surge Modeling for Urban Mobility


## 1) Project overview
This project predicts:
- **Ride price** (regression)
- **Whether surge pricing is active** (classification)

It uses **historical Uber/Lyft ride data** and **weather conditions** (Boston, 2018) and provides an interactive **Streamlit** web application to test predictions with simple inputs.

## 2) What you get from this project
- A complete data workflow: load, clean, align ride data with weather data, and build features
- Two machine learning models:
  - **Price prediction** using `XGBRegressor`
  - **Surge prediction** using `XGBClassifier`
- A Streamlit app with pages for:
  - Price prediction
  - Surge prediction
  - Data exploration and visualizations
  - Model performance summaries

## 3) Dataset details
This repository uses two CSV files:
- `cab_rides.csv`: ride-level information (cab type, service name, pickup, dropoff, distance, timestamp, price, surge multiplier)
- `weather.csv`: weather observations over time (temperature, rain, wind, humidity, clouds, pressure, timestamp)

Important notes:
- The data is from **Boston** and is limited to **2018** (commonly Nov–Dec in many versions of this dataset).
- The app UI uses a date range that matches the dataset to avoid invalid inputs.

## 4) Project structure
```text
Traffic Prediction & Urban Mobility Analysis/
├── streamlit_app.py
├── Traffic Prediction & Urban Mobility.ipynb
├── cab_rides.csv
├── weather.csv
├── run_streamlit.bat
├── run_streamlit.ps1
├── data workflow/
│   ├── Preprocessing.excalidraw.png
│   └── Workflow.excalidraw
└── README.md
```

## 5) Modeling approach

### 5.1 Price prediction (regression)
- **Model**: XGBoost Regressor (`XGBRegressor`)
- **Target**: `price`
- **Typical metrics**: MAE, RMSE, R²

Data leakage prevention:
- `surge_multiplier` is **excluded** from the regression features because it is strongly related to the price and can leak information into the model.

### 5.2 Surge prediction (classification)
- **Model**: XGBoost Classifier (`XGBClassifier`)
- **Target**: `is_surge` where:
  - `is_surge = 1` if `surge_multiplier > 1`
  - `is_surge = 0` otherwise

Imbalance handling:
- Uses strategies like `scale_pos_weight` and decision threshold tuning (implemented in the notebook/app) to improve results for the minority class.

## 6) Feature engineering (high level)
The project builds features that reflect pricing behavior and demand patterns:
- **Time-based features**: hour, day of week, weekend flag, rush hour flag
- **Weather features**: aggregated to hourly level to match ride timestamps
- **Location features**:
  - geographic distance between pickup and dropoff (haversine)
  - frequency encoding for popular pickup/dropoff locations
- **Categorical encoding**:
  - one-hot encoding for `cab_type` and `name` (service type)

## 7) How to run the Streamlit app

### 7.1 Install requirements
From the project folder, install dependencies:
```bash
pip install streamlit pandas numpy scikit-learn xgboost matplotlib seaborn
```

Optional (if you use SHAP in notebook work):
```bash
pip install shap
```

### 7.2 Start the app (recommended)
You can run Streamlit in any of these ways:

Option A (direct):
```bash
streamlit run streamlit_app.py
```

Option B (Windows scripts):
- Double-click `run_streamlit.bat`
or run:
```powershell
.\run_streamlit.ps1
```

### 7.3 Open in browser
Streamlit will print a local URL (usually):
```text
http://localhost:8501
```

## 8) Using the app (simple inputs)
The app is designed so you do not need to guess valid values.

Recommendations:
- Use the **preset buttons** (Default / Rush Hour / Weekend / Rainy Day) to fill reasonable values automatically.
- Keep the date inside the dataset range shown in the UI.
- If you are unsure about weather values, keep the default values.

## 9) Common errors and how to fix them

### 9.1 Feature mismatch error (most common)
You may see an error like:
> Feature names should match those that were passed during fit

This usually happens when:
- The prediction input columns do not exactly match the model training columns
- A preprocessing step creates extra columns or misses required columns

How the app avoids this:
- It aligns the input features to the exact training feature list (the app builds a full feature vector and fills missing columns with zeros).

If you still see it:
1. Refresh the page
2. Use a preset (Default) to fill all fields
3. Do not leave any input empty

### 9.2 Date/time outside dataset
If you choose dates outside the dataset range, predictions can be unstable or the feature creation logic may not match the training distribution.

Fix:
- Use the date range shown in the app (2018 range).

## 10) Limitations
- Historical dataset only (Boston, 2018)
- Predictions are estimates based on patterns in the dataset
- No real-time traffic, holidays, concerts, sports events, or live demand signals

## 11) Future improvements
- Add real traffic and holiday/event features
- Hyperparameter tuning and cross-validation for stronger generalization
- Model versioning and experiment tracking
- Support additional cities and time periods

## 12) Tech stack
- Python
- Pandas, NumPy
- scikit-learn
- XGBoost
- Streamlit
- Matplotlib, Seaborn