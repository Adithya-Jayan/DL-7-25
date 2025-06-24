#Import necessary packages.
#import tensorflow_hub as hub
import os
import re
import torch
import pandas as pd
import numpy as np
import tensorflow_hub as hub
import yfinance as yf
from datetime import datetime
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# -------------------------------------------------------------------------
#  Web Scraping Imports
# -------------------------------------------------------------------------
from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
from datetime import timedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
# -----------------------------------------------------------------------------------
# Function to get the latest news data from various sources
# This function combines the latest articles from BullionVault, Yahoo Finance, and Reuters.
def extract_news_data_old():
    bullion_df = get_latest_bullionvault_articles()
    yf_df=get_latest_yf_articles()
    yf_df['Date']=pd.to_datetime(yf_df['Date'],errors='coerce').dt.date
    reuters_df = get_reuters_articles()
    three_days_ago = pd.to_datetime('today').date() - timedelta(days=3)

    df_combined = pd.concat([bullion_df, yf_df, reuters_df], ignore_index=True)
    df_combined = df_combined.sort_values(by='Date')
    df_combined=df_combined[df_combined['Date'] >= three_days_ago]
    # Placeholder for actual news data extraction logic
    return df_combined

def extract_news_data(local_news=False):
    bullion_df = get_latest_bullionvault_articles()
    yf_df=get_latest_yf_articles()
    yf_df['Date']=pd.to_datetime(yf_df['Date'], format='mixed', utc=True).dt.date
    reuters_df = get_reuters_articles()
    df_list_for_concatenation = [bullion_df, yf_df, reuters_df]
    if local_news:
        telugu_news_df=fetch_bbc_telugu_news()
        df_list_for_concatenation.append(telugu_news_df)
    
    three_days_ago = pd.to_datetime('today').date() - timedelta(days=3)

    df_combined = pd.concat(df_list_for_concatenation, ignore_index=True)
    df_combined = df_combined.sort_values(by='Date')
    df_combined=df_combined[df_combined['Date'] >= three_days_ago]
    return df_combined

# -----------------------------------------------------------------------------------
# Function to clean and prepare articles for tokenizer
def clean_and_prepare_articles(
    input_file="bullionvault_articles.csv",
    output_file="tokenizer_ready_output.csv",
    max_chunk_len=1500
):
    # ===== STEP 1: Load input file =====
    df = pd.read_csv(input_file)
    df.columns = [c.strip() for c in df.columns]

    # ===== STEP 2: Detect and prepare content column =====
    if 'News' in df.columns:
        df['Text'] = df['News']
    elif 'Content' in df.columns:
        df['Text'] = df['Content']
    else:
        raise Exception("no valid text column found (check if column is named News or Content)")

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    elif 'Dates' in df.columns:
        df['Date'] = pd.to_datetime(df['Dates'], errors='coerce')
    else:
        df['Date'] = pd.NaT

    # ===== STEP 3: Basic cleaning =====
    def clean_text(t):
        if not isinstance(t, str):
            return ""
        t = t.replace("Title:", "")
        t = t.strip()
        return ' '.join(t.split())

    df['Text'] = df['Text'].apply(clean_text)
    df = df[df['Text'] != ""]

    # ===== STEP 4: Break long content into chunks =====
    def split_into_sentences(txt):
        return re.split(r'(?<=[.!?])\s+(?=[A-Z])', txt)

    def chunk_text(text, max_len=max_chunk_len):
        sents = split_into_sentences(text)
        result = []
        chunk = ""
        for s in sents:
            if len(chunk) + len(s) + 1 <= max_len:
                chunk += " " + s if chunk else s
            else:
                result.append(chunk)
                chunk = s
        if chunk:
            result.append(chunk)
        return result

    # ===== STEP 5: Sentiment labeling =====
    def get_label(text):
        text = text.lower()
        if any(word in text for word in ['gold prices rise', 'surge', 'bullish', 'rate cut', 'safe-haven', 'investing trends higher']):
            return 0
        elif any(word in text for word in ['fall', 'drop', 'bearish', 'loss', 'decline', 'sell-off']):
            return 2
        else:
            return 1

    if 'Price Sentiment' in df.columns:
        convert = {"positive": 0, "neutral": 1, "none": 1, "negative": 2}
        df['Label'] = df['Price Sentiment'].str.lower().map(convert)
    else:
        df['Label'] = df['Text'].apply(get_label)

    df = df.dropna(subset=['Label'])

    # ===== STEP 6: Build final rows =====
    final_rows = []
    for idx, row in df.iterrows():
        text = row['Text']
        date = row['Date']
        label = int(row['Label'])
        parts = chunk_text(text)
        for p in parts:
            line = clean_text(p.lower())
            if line:
                final_rows.append({
                    'date': date,
                    'text': line,
                    'label': label
                })

    # ===== STEP 7: Save the final cleaned and labeled file =====
    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(output_file, index=False)
    print("Done! Saved to:", output_file)

# -----------------------------------------------------------------------------------

# Function to get the latest articles from BullionVault
# This function scrapes the latest articles from BullionVault's news section.
def get_latest_bullionvault_articles(URL="https://www.bullionvault.com/gold-news"):
    response = requests.get(URL)
    soup = BeautifulSoup(response.content, 'html.parser')
    latest=soup.find(id='views-bootstrap-grid-1').find_all(class_='field-content')
    list_data = []
    for item in latest:
        date=item.find(class_='views-field-created')
        if not date:
            continue
        link=item.find(class_='views-field-title').find('a')['href']
        page_response = requests.get(link)
        page_soup = BeautifulSoup(page_response.content, 'html.parser')
        content = page_soup.find('div', class_='field field-name-body field-type-text-with-summary field-label-hidden')
        title = page_soup.find('h1').text.strip()
        content_text = content.text.strip() if content else ''
        data_point = {'Date': date.text.strip() if date else 'N/A', 'Content': title + ':' + content_text}
        list_data.append(data_point)
    list_df=pd.DataFrame(list_data)
    list_df['Date']= pd.to_datetime(list_df['Date'],errors='coerce').dt.date
    return list_df

# Function to get the latest articles from Yahoo Finance
# This function scrapes the latest articles from Yahoo Finance's news section.
def yf_extract_info(item):
    link=item.find('a',class_='subtle-link')['href']
    title=item.find('a',class_='subtle-link')['title']

    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration

    page_driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    page_driver.get(link)

    page_soup = BeautifulSoup(page_driver.page_source, 'html.parser')
    content = page_soup.find('div', class_='body')
    content_text = content.text.strip() if content else ''
    date= page_soup.find('div', class_= lambda c: c and c.startswith("byline")).find('time')
    data_point = {'Date': date.text.strip() if date else 'N/A', 'Content': title + ':' + content_text}
    page_driver.quit()
    return data_point

def get_latest_yf_articles(URL="https://finance.yahoo.com/news/"):
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.get(URL)
    time.sleep(2)
    last_height = driver.execute_script("return document.body.scrollHeight")
    count = 0
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # Wait for new content to load
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        count += 1
        if count > 1:
            break
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    articles = soup.find_all('li', class_='story-item')
    list_data = []
    #print(f"Found {len(articles)} articles on Yahoo Finance.")
    for article in articles:
        try:
            list_data.append(yf_extract_info(article))
        except Exception as e:
            continue
    driver.quit()
    list_df=pd.DataFrame(list_data)
    list_df['Date'].dropna(inplace=True)
    #list_df['Date']= pd.to_datetime(list_df['Date'],errors='coerce').dt.date
    return list_df

# Function to get the latest articles from Reuters
# This function scrapes the latest articles from Reuters' news section.
def get_reuters_article_text(item, base_URL="https://www.reuters.com"):
    title = item.get_text(strip=True)
    link = item.find('a', href=True)['href']
    if not link.startswith('http'):
        link = base_URL + link
    chrome_options = Options()
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(link)
    time.sleep(2)
    page_soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.quit()
    article = page_soup.find_all("div", class_=lambda c: c and c.startswith("article-body__paragraph"))
    article_text = ""
    for para in article:
        paragraph_text = para.get_text(strip=True)
        article_text = article_text + "." + paragraph_text
    date = page_soup.find("span", class_=lambda c: c and c.startswith("date-line__date")).get_text(strip=True)
    data_point = {'Date': date, 'Content': title + ':' + article_text}
    return data_point

def get_reuters_articles_list(URL):
    chrome_options = Options()
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(URL)
    time.sleep(2)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.quit()
    articles=soup.find_all("div", class_=lambda c: c and c.startswith("story-card__area-headline"))
    list_data = []
    for article in articles:
        try:
            list_data.append(get_reuters_article_text(article))
        except Exception as e:
            continue
    list_df=pd.DataFrame(list_data)
    list_df['Date']= pd.to_datetime(list_df['Date'],errors='coerce').dt.date
    return list_df

def get_reuters_articles():
    base_URL="https://www.reuters.com"
    search_query="/site-search/?query=gold"
    df = pd.DataFrame(columns=['Date', 'Content'])
    for section_val in ['all']:
        for offset_nb in range(0, 40, 20):
            offset =f"&offset={offset_nb}"
            section=f"&section={section_val}"
            URL = base_URL + search_query + offset + section
            try:
                df_latest=get_reuters_articles_list(URL)
                df = pd.concat([df, df_latest], ignore_index=True)
            except Exception as e:
                print(f"Error fetching articles from {URL}: {e}")
                continue
    return df

def sarvam_translate_text(text, source_lang='te-IN', target_lang='en-IN'):
    """
    Translate text using Sarvam API
    """
    url = "https://api.sarvam.ai/translate"
    payload = {
        "input": text,
        "source_language_code": source_lang,
        "target_language_code": target_lang
    }
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    try:
        dotenv.load_dotenv()
        SARVAM_API_KEY= os.getenv('SARVAM_API_KEY')
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        translated_text = response.json()['translated_text']
        return translated_text
    except requests.exceptions.RequestException as e:
        print(f"Error during translation: {e}")
        return text
    
def fetch_bbc_telugu_news(URL="https://www.bbc.com/telugu/popular/read"):
    """
    Fetches the latest news from BBC Telugu.
    """
    try:
        response = requests.get(URL)
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = soup.find('ol').find_all('a')
        list_data = []
        for article in articles:
            content = article.get_text(strip=True)+":"
            link = article['href']
            page_response = requests.get(link)
            page_soup = BeautifulSoup(page_response.content, 'html.parser')
            main_tag = page_soup.find('main')
            date=main_tag.find('time')
            if date:
                date=date["datetime"]
            for script in main_tag(['script', 'style']):
                script.decompose()
            for figure in main_tag.find_all('figure'):
                figure.decompose()
            #for section in main_tag.find_all('section'):
            #    section.decompose()
            paragraphs = main_tag.find_all('p')
            for p in paragraphs:
                if len(content) + len(p.get_text(strip=True)) + 1 <= 1000:
                    content += p.get_text(strip=True)
            translated_content = sarvam_translate_text(content)
            #translated_content = content  # Use original content for now
            data_point = {'Date': date, 'Content': translated_content}
            list_data.append(data_point)
            list_df=pd.DataFrame(list_data)
            list_df['Date']= pd.to_datetime(list_df['Date'],errors='coerce').dt.date
        return list_df
    except Exception as e:
        return
# ---------------------------------------------------------------------

# Function to predict sentiment using a pre-trained model
# ---------------------------------------------------------------------
def predict_sentiment(text, model, tokenizer, device):
    """
    Predict sentiment for a given text using the trained model
    Returns: Dictionary containing prediction results including logits
    """
    try:
        # Prepare model
        model.eval()
        model = model.to(device)

        # Tokenize input
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=1)[0].item()

        # Map prediction to sentiment
        sentiment_map = {0: "positive", 1: "neutral", 2: "negative"}
        confidence = probs[0][predicted_class].item()

        return {
            "text": text,
            "sentiment": sentiment_map[predicted_class],
            "confidence": f"{confidence:.4f}",
            "logits": logits[0].cpu().numpy().tolist(),
            "probabilities": {
                "positive": f"{probs[0][0].item():.4f}",
                "neutral": f"{probs[0][1].item():.4f}",
                "negative": f"{probs[0][2].item():.4f}"
            }
        }

    except Exception as e:
        print(f"Error in prediction: {e}")
        return None
    
# Function to batch predict sentiment and update CSV file
# ---------------------------------------------------------------------
def batch_predict_and_update_csv( csv_path, model_path, output_path):
    # Load data
    df = pd.read_csv(csv_path)
    if "text" not in df.columns:
        raise ValueError("CSV must have a 'text' column.")

    # Load model and tokenizer
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Prepare columns for results
    sentiments = []
    confidences = []
    logits_list = []
    probabilities_list = []

    # Predict for each row
    for text in df["text"]:
        result = predict_sentiment(text, model, tokenizer, device)
        if result:
            sentiments.append(result["sentiment"])
            confidences.append(result["confidence"])
            logits_list.append(result["logits"])
            probabilities_list.append(result["probabilities"])
        else:
            sentiments.append(None)
            confidences.append(None)
            logits_list.append(None)
            probabilities_list.append(None)

    # Add results to DataFrame
    df["predicted_sentiment"] = sentiments
    df["sentiment_confidence"] = confidences
    df["sentiment_logits"] = logits_list
    df["sentiment_probabilities"] = probabilities_list

    # Save updated CSV
    df.to_csv(output_path, index=False)
    print(f"Updated file saved to {output_path}")

# Load the model from a checkpoint
# ---------------------------------------------------------------------
def load_checkpoint(filepath, model, device='cpu'):
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)  # Set weights_only=False
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

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
    #print(gold.head())
    return gold

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


# Technical Indicator Calculation
# ---------------------------------------------------------------------
def add_technical_indicators(gold: pd.DataFrame) -> pd.DataFrame:
    # Technical indicators
    print("Adding technical indicators...")

    gold['Returns'] = gold['Close'].pct_change()
    gold['MA_5'] = gold['Close'].rolling(window=5).mean()
    gold['MA_20'] = gold['Close'].rolling(window=20).mean()
    gold['MA_50'] = gold['Close'].rolling(window=50).mean()
    gold['Volatility'] = gold['Returns'].rolling(window=20).std()
    gold['RSI'] = calculate_rsi(gold['Close'])

    print("Calculating Bollinger Bands...")
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

    print(f"Added indicators to {len(gold)} rows.")

    # Save full DataFrame with indicators
    gold.to_csv("Data/GOLDBEES_ETF_price_data_technical_indicators.csv")
    print("Saved technical indicators to Data/GOLDBEES_ETF_price_data_technical_indicators.csv")

    return gold

#Add continuous sentiment Based on Price Trend with Labels
# ---------------------------------------------------------------------
def generate_sentiment_from_trend_with_labels(gold: pd.DataFrame, sentiment_today: float = 0.0, seed: int = 42) -> pd.DataFrame:
    """
    Generate numeric sentiment scores and sentiment labels based on price returns.

    Args:
        gold (pd.DataFrame): DataFrame with 'Close' column
        seed (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: Updated DataFrame with 'Sentiment' and 'Sentiment_Label' columns
    """
    import random
    random.seed(seed)

    gold = gold.copy()
    gold['Returns'] = gold['Close'].pct_change()

    sentiment_scores = []
    sentiment_labels = []

    for ret in gold['Returns']:
        if pd.isna(ret):
            sentiment = 0.0
        elif ret > 0.01:
            sentiment = round(random.uniform(0.5, 1.0), 2)
        elif ret > 0.0:
            sentiment = round(random.uniform(0.1, 0.5), 2)
        elif ret > -0.01:
            sentiment = round(random.uniform(-0.5, -0.1), 2)
        else:
            sentiment = round(random.uniform(-1.0, -0.5), 2)

        # Assign label
        if sentiment > 0.1:
            label = 'positive'
        elif sentiment < -0.1:
            label = 'negative'
        else:
            label = 'neutral'

        sentiment_scores.append(sentiment)
        sentiment_labels.append(label)

    gold['Sentiment'] = sentiment_scores
    gold['Sentiment_Label'] = sentiment_labels

    # Inject today's sentiment into the last row
    #gold['Sentiment'] = 0.0  # Initialize all with neutral
    if not gold.empty:
        gold.iloc[-1, gold.columns.get_loc('Sentiment')] = sentiment_today
        print(f"Injected today's sentiment ({sentiment_today:+.2f}) into the last row.")

    # Save to CSV
    os.makedirs("Data", exist_ok=True)
    gold.to_csv("Data/GOLDBEES_ETF_price_data_technical_indicators_sentiment.csv")
    print("Sentiment columns added and saved to Data/GOLDBEES_ETF_price_data_technical_indicators_sentiment.csv")

    return gold

#Preprocess the dataset
def preprocess_dataset(df_raw):

    #Create a copy before processing
    df_processed = df_raw.copy()

    #Parse prediction text into a dictionary
    df_processed['sentiment_probabilities'] = df_processed['sentiment_probabilities'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    #Extract net sentiment score [Between -1 and 1]
    df_processed['sentiment'] = df_processed['sentiment_probabilities'].apply(lambda x: float(x['positive']) - float(x['negative']))
    df_processed['sentiment'] = df_processed['sentiment'] * df_processed['sentiment_confidence']

    df_out = df_processed[['date', 'text', 'sentiment']].copy()


    return df_out


def generate_topic_encodings(df_data):
    """
    Generate topic encodings for the text data in the DataFrame using the Universal Sentence Encoder.
    
    Parameters:
    df_data (DataFrame): DataFrame containing the text data.
    
    Returns:
    DataFrame: DataFrame with an additional column for topic encodings.
    """
    
    # Load the Universal Sentence Encoder model
    model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"  # This is around 1 GB in size, it took a while for me to run this.
    embed = hub.load(model_url)

    # Generate embeddings
    embeddings = embed(list(df_data['text']))

    # Add it to our DataFrame
    df_data['topic_encodings'] = list(np.array(embeddings))
    
    return df_data


def get_sentiment_combined_encodings(df_data):

    df_data_premerge = df_data.copy()
    df_data_premerge['Date'] = pd.to_datetime(df_data_premerge['date'])

    # Merge topic_encodings and sentiment
    df_data_premerge['sentiment_combined_encodings'] = df_data_premerge['topic_encodings'] * df_data_premerge['sentiment']

    final_df = df_data_premerge[['sentiment_combined_encodings']].copy()

    return final_df

def add_gold_price_change_with_weekend_handling_old(df_data, df_gold):
    """
    Merge news data with gold price changes, handling weekends by carrying forward Friday's price for Saturday/Sunday news.
    """
    df_gold_premerge = df_gold.copy()
    df_data_premerge = df_data.copy()

    df_gold_premerge['Date'] = pd.to_datetime(df_gold_premerge['Date'])
    df_data_premerge['Date'] = pd.to_datetime(df_data_premerge['date'])

    # Compute the relative change in price from one day to the next before merging.
    df_gold_premerge = df_gold_premerge.sort_values(by='Date').reset_index(drop=True)
    df_gold_premerge['next_day_price'] = df_gold_premerge['Close'].shift(-1)
    df_gold_premerge['next_day'] = df_gold_premerge['Date'].shift(-1)
    df_gold_premerge['day_gap'] = (df_gold_premerge['next_day'] - df_gold_premerge['Date']).dt.days
    df_gold_premerge['relative_change'] = (df_gold_premerge['next_day_price'] - df_gold_premerge['Close']) / df_gold_premerge['day_gap']
    df_gold_premerge['relative_change'] = 100 * (df_gold_premerge['relative_change'] / df_gold_premerge['Close'])

    # Adjust Date to be the day before, as we want to predict next day's price
    df_gold_premerge['Date'] = df_gold_premerge['Date'] - pd.Timedelta(days=1)

    # Create a mapping from date to price change
    gold_map = df_gold_premerge.set_index('Date')['relative_change'].to_dict()

    # For each news date, find the most recent available gold price change (carry forward for weekends)
    def get_price_change(news_date):
        # If news_date is in gold_map, return directly
        if news_date in gold_map:
            return gold_map[news_date]
        # If not, go backwards to find the last available date (for weekends)
        prev_date = news_date
        while prev_date not in gold_map or pd.isna(gold_map.get(prev_date, None)):
            prev_date -= pd.Timedelta(days=1)
            # To avoid infinite loop, break if too far in the past
            if prev_date < min(gold_map.keys()):
                return np.nan
        return gold_map[prev_date]

    df_data_premerge['price_percentage_change'] = df_data_premerge['Date'].apply(get_price_change)

    # Merge topic_encodings and sentiment
    df_data_premerge['sentiment_combined_encodings'] = df_data_premerge['topic_encodings'] * df_data_premerge['sentiment']

    final_df = df_data_premerge[['Date', 'text', 'sentiment', 'topic_encodings', 'sentiment_combined_encodings', 'price_percentage_change']].copy()

    return final_df

def add_gold_price_change_with_weekend_handling(df_data, df_gold):
    """
    Merge news data with gold price changes, handling weekends by carrying forward Friday's price for Saturday/Sunday news.
    """
    df_gold_premerge = df_gold.copy()
    df_data_premerge = df_data.copy()

    df_gold_premerge['Date'] = pd.to_datetime(df_gold_premerge['Date'])
    df_data_premerge['Date'] = pd.to_datetime(df_data_premerge['date'])

    # Compute the relative change in price from one day to the next before merging.
    df_gold_premerge = df_gold_premerge.sort_values(by='Date').reset_index(drop=True)
    df_gold_premerge['next_day_price'] = df_gold_premerge['Close'].shift(-1)
    df_gold_premerge['next_day'] = df_gold_premerge['Date'].shift(-1)
    df_gold_premerge['day_gap'] = (df_gold_premerge['next_day'] - df_gold_premerge['Date']).dt.days
    df_gold_premerge['relative_change'] = (df_gold_premerge['next_day_price'] - df_gold_premerge['Close']) / df_gold_premerge['day_gap']
    df_gold_premerge['relative_change'] = 100 * (df_gold_premerge['relative_change'] / df_gold_premerge['Close'])

    # Adjust Date to be the day before, as we want to predict next day's price
    df_gold_premerge['Date'] = df_gold_premerge['Date'] - pd.Timedelta(days=1)

    # Create a mapping from date to price change
    gold_map = df_gold_premerge.set_index('Date')['relative_change'].to_dict()

    # For each news date, find the most recent available gold price change (carry forward for weekends)
    def get_price_change(news_date):
        # If news_date is in gold_map, return directly
        if news_date in gold_map:
            return gold_map[news_date]
        # If not, go backwards to find the last available date (for weekends)
        prev_date = news_date
        while prev_date not in gold_map or pd.isna(gold_map.get(prev_date, None)):
            prev_date -= pd.Timedelta(days=1)
            # To avoid infinite loop, break if too far in the past
            if prev_date < min(gold_map.keys()):
                return np.nan
        return gold_map[prev_date]

    df_data_premerge['price_percentage_change'] = df_data_premerge['Date'].apply(get_price_change)

    # Merge topic_encodings and sentiment
    df_data_premerge['sentiment_combined_encodings'] = df_data_premerge['topic_encodings'] * df_data_premerge['sentiment']

    final_df = df_data_premerge[['Date', 'text', 'sentiment', 'topic_encodings', 'sentiment_combined_encodings', 'price_percentage_change']].copy()

    return final_df

def add_gold_price_change(df_data,df_gold):
    

    #Merge above dataframe with our gold data.

    df_gold_premerge = df_gold.copy()
    df_data_premerge = df_data.copy()

    df_gold_premerge['Date'] = pd.to_datetime(df_gold_premerge['Date'])
    df_data_premerge['Date'] = pd.to_datetime(df_data_premerge['date'])

    # Compute the relative change in price from one day to the next before merging.
    df_gold_premerge = df_gold_premerge.sort_values(by='Date').reset_index(drop=True)
    df_gold_premerge['next_day_price'] = df_gold_premerge['Close'].shift(-1)
    df_gold_premerge['next_day'] = df_gold_premerge['Date'].shift(-1)
    df_gold_premerge['day_gap'] = (df_gold_premerge['next_day'] - df_gold_premerge['Date']).dt.days
    df_gold_premerge['relative_change'] = (df_gold_premerge['next_day_price'] - df_gold_premerge['Close']) / df_gold_premerge['day_gap']

    #Make it a percentage change
    df_gold_premerge['relative_change'] = 100 * (df_gold_premerge['relative_change'] / df_gold_premerge['Close'])

    # # We want to predict gold price for the next day. Data to use for prediction is the day before. ]
    df_gold_premerge['Date'] = df_gold_premerge['Date'] - pd.Timedelta(days=1)

    # Perform the merge on the adjusted date
    merged_df = pd.merge(df_data_premerge, df_gold_premerge, on='Date', how='inner')
    merged_df['sentiment_combined_encodings'] =  merged_df['topic_encodings'] * merged_df['sentiment']

    #Rename relative_change column to price_percentage_change
    merged_df = merged_df.rename(columns={'relative_change':'price_percentage_change'})
    final_df = merged_df[['Date','text','sentiment','topic_encodings','sentiment_combined_encodings','price_percentage_change']].copy()

    return final_df

def group_into_variable_sets(df, max_articles_per_day=None):
    """
    Group data into variable-sized sets with padding and masking
    """
    # Group by date and get actual lists
    grouped = df.groupby('Date')['sentiment_combined_encodings'].apply(list)
    price_changes = df.groupby('Date').first()['price_percentage_change'].values
    
    # Find max set size if not provided
    if max_articles_per_day is None:
        max_articles_per_day = max(len(articles) for articles in grouped)
        print(f"Max articles per day: {max_articles_per_day}")
    
    # Pad sequences and create masks
    padded_encodings = []
    masks = []
    
    for articles in grouped:
        current_length = len(articles)
        
        # Pad with zeros if needed
        if current_length < max_articles_per_day:
            # Assuming articles are numpy arrays or lists of same dimension
            encoding_dim = len(articles[0]) if current_length > 0 else 768  # Default embedding dim
            padded = articles + [np.zeros(encoding_dim)] * (max_articles_per_day - current_length)
        else:
            padded = articles[:max_articles_per_day]  # Truncate if too long
            current_length = max_articles_per_day
        
        padded_encodings.append(padded)
        
        # Create mask: 1 for real data, 0 for padding
        mask = [1] * min(current_length, max_articles_per_day) + [0] * max(0, max_articles_per_day - current_length)
        masks.append(mask)
    
    encodings = np.array(padded_encodings, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)
    price_changes = np.array(price_changes, dtype=np.float32).reshape(-1, 1)
    
    return encodings, price_changes, masks