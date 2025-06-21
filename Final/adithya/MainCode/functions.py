#Import necessary packages.
import tensorflow_hub as hub
import pandas as pd
import numpy as np


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

def group_into_set(df,articles_per_day):

    # Create dataset and dataloader
    encodings = df.groupby('Date')['sentiment_combined_encodings'].apply(list).apply(lambda x: [x[i%len(x)] for i in range(articles_per_day)])
    price_percentage_changes = df.groupby('Date').first()['price_percentage_change'].values

    encodings = np.array(encodings.tolist(), dtype=np.float32)
    price_percentage_changes = np.array(price_percentage_changes, dtype=np.float32).reshape(-1, 1)  # Ensure shape is (N, 1)

    return(encodings, price_percentage_changes)