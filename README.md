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
   - tensorflow (or keras for TF backend)
   - streamlit

   *Snowflake Streamlit note:* TensorFlow is not currently available in the Snowflake package repository. The app now detects this automatically and falls back to a lightweight scikit-learn multilayer perceptron for predictions. To use that path, omit TensorFlow from your Snowflake package list.

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
