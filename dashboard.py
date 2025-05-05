import streamlit as st
from sentiment_model import load_sentiment_model, analyze_sentiment
from topic_model import train_lda_model, get_topics

st.title("AI-Powered Sentiment Analysis and Topic Modeling")

# Sentiment Analysis
st.header("Sentiment Analysis")
text = st.text_area("Enter text for sentiment analysis")
if st.button("Analyze Sentiment"):
    sentiment_model = load_sentiment_model()
    sentiment = analyze_sentiment(text, sentiment_model)
    st.write(sentiment)

# Topic Modeling
st.header("Topic Modeling")
texts = st.text_area("Enter multiple texts (separated by line breaks)")
if st.button("Extract Topics"):
    texts_list = texts.split("\n")
    lda_model, dictionary, corpus = train_lda_model(texts_list)
    topics = get_topics(lda_model)
    st.write(topics)
