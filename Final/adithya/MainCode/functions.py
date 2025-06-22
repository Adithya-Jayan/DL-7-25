#Import necessary packages.
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import torch


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


def load_checkpoint(filepath, model, optimizer,device):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    val_loss = checkpoint['val_loss']
    return epoch, loss, model,optimizer