import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from finetuning.finbert_finetune_refactored import predict_sentiment

def batch_predict_and_update_csv(
    csv_path="Final_gold-dataset-sinha-khandait1.csv",
    model_path="./finbert_best_model_merged",
    output_path="Final_gold-dataset-sinha-khandait1_with_predictions.csv"
):
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

if __name__ == "__main__":
    batch_predict_and_update_csv()