#  Imports
# ---------------------------------------------------------------------
import os  # For folder creation
import yfinance as yf
from datetime import datetime
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Download historical GOLDBEES ETF price data
# ---------------------------------------------------------------------
def download_gold_prices(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    # Download historical GOLDBEES ETF price data
    print("\nStep 1: Downloading gold price data (GOLDBEES.BO)...")
    gold = yf.download('GOLDBEES.BO', start=start_date, end=end_date, progress=False)
    print("Download complete.")
    #print(gold.head())

    if isinstance(gold.columns, pd.MultiIndex):
        gold.columns = gold.columns.get_level_values(0)

    gold = gold[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    gold.columns.name = None  # Remove "Price" label from column index

    print("Current working directory:", os.getcwd())  
    # Ensure the 'Data' directory exists
    os.makedirs("Data", exist_ok=True)

    # Save raw data to CSV
    gold.to_csv("Data/GOLDBEES_ETF_price_data.csv")
    print("Saved raw gold price data to Data/GOLDBEES_ETF_price_data.csv")
    #print(gold.columns)
    print(gold.head())
    return gold

# ---------------------------------------------------------------------
# Technical Indicator Calculation
# ---------------------------------------------------------------------

#RSI Calculation
# ---------------------------------------------------------------------
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    # Relative Strength Index calculation
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ---------------------------------------------------------------------
# Technical Indicator Calculation
# ---------------------------------------------------------------------
def add_technical_indicators(gold: pd.DataFrame) -> pd.DataFrame:
    # Technical indicators
    print("\nStep 2: Adding technical indicators...")

    gold['Returns'] = gold['Close'].pct_change()
    gold['MA_5'] = gold['Close'].rolling(window=5).mean()
    gold['MA_20'] = gold['Close'].rolling(window=20).mean()
    gold['MA_50'] = gold['Close'].rolling(window=50).mean()
    gold['Volatility'] = gold['Returns'].rolling(window=20).std()
    gold['RSI'] = calculate_rsi(gold['Close'])

    print("\nStep 3: Calculating Bollinger Bands...")
    rolling_std = gold['Close'].rolling(window=20).std()
    gold['BB_upper'] = gold['MA_20'] + (rolling_std * 2)
    gold['BB_lower'] = gold['MA_20'] - (rolling_std * 2)
    gold['BB_width'] = gold['BB_upper'] - gold['BB_lower']
    gold['BB_position'] = (gold['Close'] - gold['BB_lower']) / gold['BB_width']

    # MACD and Signal Line
    exp1 = gold['Close'].ewm(span=12, adjust=False).mean()
    exp2 = gold['Close'].ewm(span=26, adjust=False).mean()
    gold['MACD'] = exp1 - exp2
    gold['MACD_Signal'] = gold['MACD'].ewm(span=9, adjust=False).mean()
    gold['MACD_Hist'] = gold['MACD'] - gold['MACD_Signal']

    # Momentum (n-day price diff)
    gold['Momentum_10'] = gold['Close'] - gold['Close'].shift(10)

    # Rate of Change (ROC)
    gold['ROC_10'] = gold['Close'].pct_change(periods=10)


    # Drop NaNs and infinite values after all calculations
    gold.replace([np.inf, -np.inf], np.nan, inplace=True)
    gold.dropna(inplace=True)


    # Desired column order
    columns_order = [
        'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Returns', 'MA_5', 'MA_20', 'MA_50', 'Volatility', 'RSI',
        'BB_upper', 'BB_lower', 'BB_width', 'BB_position',
        'MACD', 'MACD_Signal', 'MACD_Hist',
        'Momentum_10', 'ROC_10'
    ]

    # Reorder and handle missing columns
    existing_cols = [col for col in columns_order if col in gold.columns]
    gold = gold[existing_cols]

    print(f"\nAdded indicators to {len(gold)} rows.")

    # Save full DataFrame with indicators
    print(gold)
    gold.to_csv("Data/GOLDBEES_ETF_price_data_technical_indicators.csv")
    print("Saved technical indicators to Data/GOLDBEES_ETF_price_data_technical_indicators.csv")

    return gold

def get_latest_gold_data_with_sentiment(news_sentiment, csv_path):
    """
    Reads the technical indicators CSV, selects the latest available row up to today,
    injects the news_sentiment, and returns the DataFrame row and exogenous features.
    """
    gold_df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')

    today = pd.Timestamp(datetime.now().date())

    if gold_df.empty:
        raise ValueError("The technical indicators CSV is empty. Cannot make prediction.")

    if today in gold_df.index:
        df = gold_df.loc[[today]]
    else:
        available_dates = gold_df.index[gold_df.index <= today]
        if len(available_dates) == 0:
            raise ValueError("No available data before or on today's date in the technical indicators CSV.")
        last_date = available_dates.max()
        df = gold_df.loc[[last_date]]

    # Define exogenous features
    exog_cols = [
        'Returns', 'MA_5', 'MA_20', 'MA_50', 'Volatility',
        'RSI', 'BB_upper', 'BB_lower', 'BB_width',
        'BB_position', 'Sentiment',
        'MACD', 'MACD_Signal', 'MACD_Hist',
        'Momentum_10', 'ROC_10'
    ]

    # Ensure the DataFrame has the required columns
    for col in exog_cols:
        if col not in df.columns:
            df[col] = 0.0

    # Add the Sentiment value to the last row
    if not df.empty:
        df.loc[df.index[-1], 'Sentiment'] = news_sentiment

    # Create the exogenous DataFrame with the last row of data
    exog = df[exog_cols].tail(1).copy()

    if exog.shape[0] == 0:
        raise ValueError("No data available for prediction. Check if the CSV contains valid rows.")

    return df, exog