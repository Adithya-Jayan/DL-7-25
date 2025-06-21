


#Function to load and preprocess the dataset

#Preprocess the dataset
def preprocess_dataset(df_raw):

    #Create a copy before processing
    df_processed = df_raw.copy()

    #Parse prediction text into a dictionary
    df_processed['sentiment_probabilities'] = df_processed['sentiment_probabilities'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    #Extract net sentiment score [Between -1 and 1]
    df_processed['sentiment'] = df_processed['sentiment_probabilities'].apply(lambda x: float(x['positive']) - float(x['negative']))
    df_processed['sentiment'] = df_processed['sentiment'] * df_processed['sentiment_confidence']

    df_out = df_processed[['date', 'text', 'label', 'sentiment']].copy()


    return df_out
