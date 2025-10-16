# Stock Price Model and Streamlit App

This project contains:
- `stock_price.ipynb` — a beginner-friendly notebook that trains an LSTM model to predict stock prices.
- `web_stock_price_predictor.py` — a Streamlit app that loads the trained model and visualizes predictions and simple forecasts.

## Quick start

1. Create a Python environment (recommended) and install dependencies:
   - yfinance
   - pandas
   - numpy
   - scikit-learn
   - matplotlib
   - streamlit

   *Optional TensorFlow backend:* Install `tensorflow-cpu==2.15.1` locally if you want to use the LSTM model. Skip this package on Snowflake (not available there); the app will switch to the built-in scikit-learn fallback automatically.

2. Open and run the notebook `stock_price.ipynb` to:
   - Download data
   - Train the LSTM model
   - Save the model as `Latest_stock_price_model.keras`

3. Launch the Streamlit app from this folder:

```powershell
# From the stock_price_model folder
streamlit run .\web_stock_price_predictor.py
```

4. In the app sidebar:
   - Choose ticker (e.g., GOOG, AAPL)
   - Adjust years, lookback window, and forecast horizon

## Notes
- The app automatically picks `Adj Close` if available, otherwise falls back to `Close`.
- Forecasts are naive iterative predictions, useful for demo — not investment advice.
