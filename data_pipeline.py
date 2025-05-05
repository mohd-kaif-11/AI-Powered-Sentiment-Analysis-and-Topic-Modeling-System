import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Clean and preprocess text data.
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

def load_and_preprocess_data(filepath):
    """
    Load and preprocess the dataset from a CSV file.
    """
    df = pd.read_csv(filepath)
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    return df

def split_data(df, target_column):
    """
    Split the data into training and testing sets.
    """
    X = df['cleaned_text']
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)
