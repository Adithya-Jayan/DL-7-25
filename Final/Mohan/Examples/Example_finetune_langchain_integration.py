import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain.tools import Tool
from finetuning.finbert_finetune_refactored import predict_sentiment  # Make sure this import works

# Define a function for LangChain to use
def sentiment_analysis_tool(text, model, tokenizer, device):
    result = predict_sentiment(text, model, tokenizer, device)
    if result:
        return (
            f"Text: {text}\n"
            f"Sentiment: {result.get('sentiment')}\n"
            f"Confidence: {result.get('confidence')}\n"
            f"Logits: {result.get('logits')}\n"
            f"Class Probabilities: {result.get('probabilities')}\n"
        )
    else:
        return "Prediction failed."

# Create a LangChain Tool
def get_sentiment_tool(model, tokenizer, device):
    return Tool(
        name="FinBERT Sentiment Analysis",
        func=lambda text: sentiment_analysis_tool(text, model, tokenizer, device),
        description="Predicts sentiment (positive, neutral, negative) for a given financial text and reports logits and probabilities."
    )

def test_model_with_langchain(model_path="../finbert_best_model_merged"):
    """
    Test the trained model on sample texts using LangChain Tool
    """
    try:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        sentiment_tool = get_sentiment_tool(model, tokenizer, device)

        test_texts = [
            "The company reported strong earnings growth and increased dividend.",
            "The stock market remained flat amid mixed economic signals.",
            "Investors lost millions as the company filed for bankruptcy."
        ]

        for text in test_texts:
            print(sentiment_tool.run(text))

    except Exception as e:
        print(f"Error in LangChain testing: {e}")

if __name__ == "__main__":
    test_model_with_langchain()