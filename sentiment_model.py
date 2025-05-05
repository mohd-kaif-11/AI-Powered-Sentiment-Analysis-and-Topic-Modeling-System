from transformers import pipeline
import pandas as pd

def load_sentiment_model():
    """
    Load a pre-trained sentiment analysis model.
    """
    return pipeline('sentiment-analysis')

def analyze_sentiment(text, model):
    """
    Analyze sentiment for a single piece of text.
    """
    return model(text)

def batch_analyze_sentiment(texts, model):
    """
    Analyze sentiment for a batch of texts.
    """
    return model(texts)

if __name__ == "__main__":
    # Example usage
    model = load_sentiment_model()
    texts = ["I love this product!", "This is the worst experience ever."]
    sentiments = batch_analyze_sentiment(texts, model)
    for text, sentiment in zip(texts, sentiments):
        print(f"Text: {text} | Sentiment: {sentiment}")
