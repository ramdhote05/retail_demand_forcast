import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import Ridge
import warnings

st.set_page_config(page_title="Retail Demand Forecast", layout="centered")

st.title("Retail Demand Forecasting")
st.write("Use the saved model pipeline to predict future sales and get inventory guidance.")

# prefer pipeline filename but fall back to legacy model if needed
preferred_paths = ["demand_pipeline.pkl", "demand_model.pkl"]
model_path = next((p for p in preferred_paths if os.path.exists(p)), None)

if model_path is None:
    st.error(
        "Saved model not found. Run `fyr.ipynb` to train or place `demand_pipeline.pkl`/`demand_model.pkl` here."
    )
    st.stop()

try:
    pipeline = joblib.load(model_path)
    st.write(f"Loaded model from {model_path}")
except Exception as e:
    st.error(f"Failed to load model ({model_path}): {e}")
    msg = str(e).lower()
    if 'xgboost' in msg or 'xgboost.dll' in msg or 'xgboosterror' in msg:
        st.warning("xgboost not available. Training a lightweight fallback Ridge model (fast)...")
        try:
            warnings.filterwarnings('ignore')
            df_local = pd.read_csv('retail_sales.csv')
            df_local['date'] = pd.to_datetime(df_local['date'])
            df_local = df_local.sort_values('date').reset_index(drop=True)
            df_local['year'] = df_local['date'].dt.year
            df_local['month_num'] = df_local['date'].dt.month
            df_local['day'] = df_local['date'].dt.day
            df_local['weekofyear'] = df_local['date'].dt.isocalendar().week.astype(int)
            df_local['weekday'] = df_local['date'].dt.weekday
            df_local['is_weekend'] = df_local['weekday'].isin([5,6]).astype(int)
            df_local['month_sin'] = np.sin(2 * np.pi * df_local['month_num'] / 12)
            df_local['month_cos'] = np.cos(2 * np.pi * df_local['month_num'] / 12)
            df_local['store_id'] = df_local['store_id'].astype('category')
            df_local['item_id'] = df_local['item_id'].astype('category')
            df_local['lag_1'] = df_local.groupby(['store_id','item_id'])['sales'].shift(1)
            df_local['lag_7'] = df_local.groupby(['store_id','item_id'])['sales'].shift(7)
            df_local['rolling_mean_7'] = (
                df_local.groupby(['store_id','item_id'])['sales'].shift(1).rolling(7).mean()
            )
            df_local = df_local.dropna().reset_index(drop=True)

            features = [
                'store_id','item_id','price','promo','weekday','month_num','year','day',
                'weekofyear','is_weekend','month_sin','month_cos','lag_1','lag_7','rolling_mean_7'
            ]

            X = df_local[features]
            y = df_local['sales']

            categorical_features = ['store_id','item_id']
            numeric_features = [f for f in features if f not in categorical_features]

            preprocessor = ColumnTransformer([
                ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features),
                ('num', StandardScaler(), numeric_features),
            ])

            pipeline = Pipeline([
                ('preprocess', preprocessor),
                ('model', Ridge(alpha=1.0))
            ])

            pipeline.fit(X, y)
            joblib.dump(pipeline, 'demand_pipeline_fallback.pkl')
            st.success('Fallback model trained and saved to demand_pipeline_fallback.pkl')
        except Exception as e2:
            st.error(f'Fallback training failed: {e2}')
            st.stop()
    else:
        st.info(
            "If the error mentions xgboost or native libraries, install xgboost or use a pipeline trained without compiled xgboost."
        )
        st.stop()

raw_data_path = "retail_sales.csv"
if not os.path.exists(raw_data_path):
    st.warning("`retail_sales.csv` not found. Manual entry is still available.")
    df = None
else:
    df = pd.read_csv(raw_data_path)
    df['date'] = pd.to_datetime(df['date'])

store_options = sorted(df['store_id'].unique()) if df is not None else [f"store_{i+1}" for i in range(5)]
item_options = sorted(df['item_id'].unique()) if df is not None else [f"item_{i+1}" for i in range(5)]

with st.form("prediction_form"):
    st.subheader("Input features")
    store_id = st.selectbox("Store ID", store_options)
    item_id = st.selectbox("Item ID", item_options)
    price = st.number_input("Price", value=100.0, min_value=0.0, step=0.5)
    promo = st.selectbox("Promo active", [0, 1])
    date = st.date_input("Prediction date")
    lag_1 = st.number_input("Sales lag 1 day", value=100.0, min_value=0.0, step=1.0)
    lag_7 = st.number_input("Sales lag 7 days", value=100.0, min_value=0.0, step=1.0)
    lag_14 = st.number_input("Sales lag 14 days", value=100.0, min_value=0.0, step=1.0)
    lag_30 = st.number_input("Sales lag 30 days", value=100.0, min_value=0.0, step=1.0)
    rolling_mean_7 = st.number_input("Rolling mean 7", value=100.0, min_value=0.0, step=0.5)
    rolling_mean_14 = st.number_input("Rolling mean 14", value=100.0, min_value=0.0, step=0.5)
    rolling_mean_30 = st.number_input("Rolling mean 30", value=100.0, min_value=0.0, step=0.5)
    rolling_std_14 = st.number_input("Rolling std 14", value=5.0, min_value=0.0, step=0.1)
    submitted = st.form_submit_button("Predict demand")

if submitted:
    month_num = date.month
    year = date.year
    day = date.day
    weekofyear = int(date.isocalendar()[1])
    weekday = date.weekday()
    is_weekend = int(weekday in (5, 6))
    month_sin = np.sin(2 * np.pi * month_num / 12)
    month_cos = np.cos(2 * np.pi * month_num / 12)

    input_df = pd.DataFrame({
        'store_id': [store_id],
        'item_id': [item_id],
        'price': [price],
        'promo': [promo],
        'weekday': [weekday],
        'month_num': [month_num],
        'year': [year],
        'day': [day],
        'weekofyear': [weekofyear],
        'is_weekend': [is_weekend],
        'month_sin': [month_sin],
        'month_cos': [month_cos],
        'lag_1': [lag_1],
        'lag_7': [lag_7],
        'lag_14': [lag_14],
        'lag_30': [lag_30],
        'rolling_mean_7': [rolling_mean_7],
        'rolling_mean_14': [rolling_mean_14],
        'rolling_mean_30': [rolling_mean_30],
        'rolling_std_14': [rolling_std_14],
    })

    prediction = pipeline.predict(input_df)[0]
    st.metric(label="Predicted Sales", value=f"{prediction:.1f}")

    if prediction > 170:
        insight = (
            "High demand expected. Increase inventory and prepare replenishment. "
            "Review marketing and distribution capacity."
        )
    elif prediction > 100:
        insight = (
            "Moderate demand expected. Maintain stock levels and monitor promo impact."
        )
    else:
        insight = (
            "Low demand expected. Avoid overstocking and consider localized pricing."
        )

    st.write("### Recommendation")
    st.write(insight)

    st.write("---")
    st.write("#### Input summary")
    st.write(input_df)

st.sidebar.header("About")
st.sidebar.write(
    "This app loads `demand_pipeline.pkl` and uses the same feature set as `fyr.ipynb` to predict sales. "
    "Run `streamlit run streamlit_app.py` to start the app."
)
