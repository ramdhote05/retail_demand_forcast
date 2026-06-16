# Retail Demand Forecasting

This project includes a demand forecasting notebook and a Streamlit app for interactive prediction.

- Live app: https://retaildemandforcast-8n93ffusdy4aqxthebxsta.streamlit.app/

## Files

- `fyr.ipynb`: Jupyter notebook with data loading, feature engineering, model comparison, and forecasting.
- `streamlit_app.py`: Streamlit app that loads `demand_pipeline.pkl` and predicts sales from user input.
- `retail_sales.csv`: Dataset used by the notebook.
- `demand_model.pkl` / `demand_pipeline.pkl`: Saved model pipeline files created by the notebook.

## Setup

1. Install required packages, for example:

```bash
pip install pandas numpy scikit-learn xgboost joblib streamlit
```

2. Run the notebook to generate `demand_pipeline.pkl`:

```bash
jupyter notebook fyr.ipynb
```

## Run the Streamlit app

Start the app (choose an available port):

```bash
streamlit run streamlit_app.py --server.port 8503
```

If the default port is in use, change `8503` to any free port (for example `8501` or `8502`).

## Notes

- The app prefers `demand_pipeline.pkl` but will fall back to `demand_model.pkl` if present.
- If loading a saved model fails due to missing native libraries (for example an `xgboost` DLL error), the app will train a lightweight Ridge-based fallback model from `retail_sales.csv` and save it as `demand_pipeline_fallback.pkl`.
- If `retail_sales.csv` is present, the app uses its store/item values to populate input selectors.
- The notebook includes time-based feature engineering, cross-validated model selection, and feature importance analysis.
