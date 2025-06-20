import pandas as pd
import numpy as np  
from Jaison.GOLD_ETF_functions import *

# Define a function to predict gold price using the loaded ARIMAX model
def arimax_gold_price_prediction(news_sentiment, arimax_model):

    # Use the helper function to get df and exog
    df, exog = get_latest_gold_data_with_sentiment(news_sentiment, csv_path="Jaison/Data/GOLDBEES_ETF_price_data_technical_indicators.csv")

    # For current price, use the last 'Close' value from df
    current_price = df['Close'].iloc[-1] if 'Close' in df.columns else np.nan

    predicted_price = arimax_model.forecast(steps=1, exog=exog).iloc[0]
    next_day_pct_change = ((predicted_price - current_price) / current_price) * 100
    next_day = df.index[-1] + pd.tseries.offsets.BDay(1)
    # Return the predicted price and next day percentage change
    return predicted_price, next_day_pct_change, next_day
