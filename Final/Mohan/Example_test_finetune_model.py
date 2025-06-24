import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Import predict_sentiment from your finbert_finetune_refactored.py
from finetuning.finbert_finetune_refactored import predict_sentiment

def test_model(model_path="../finbert_best_model_merged"):
    """
    Test the trained model on sample texts and print sentiment, logits, and probabilities.
    """
    try:
        # Load model and tokenizer
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Sample texts for testing
        test_texts = [
            "Dec. gold climbs $9.40, or 0.7%, to settle at $1,356.90/oz",
            "gold prices rebound rs 350 on global cues, weak rupee",
            "Gold futures down at Rs 30,244 ",
            "gold, oil trade lower as jobs data weigh"
        ]

        # Make predictions
        results = []
        for text in test_texts:
            prediction = predict_sentiment(text, model, tokenizer, device)
            if prediction:
                results.append(prediction)
                print("\nText:", text)
                print("Sentiment:", prediction["sentiment"])
                print("Confidence:", prediction["confidence"])
                print("Logits:", prediction["logits"])
                print("Class Probabilities:", prediction["probabilities"])

        return results

    except Exception as e:
        print(f"Error in testing: {e}")
        return None

# Example usage:
if __name__ == "__main__":
    test_model()