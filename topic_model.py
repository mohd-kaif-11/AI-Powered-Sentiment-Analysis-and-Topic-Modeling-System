from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

def preprocess_topic_texts(texts):
    """
    Tokenize and preprocess texts for topic modeling.
    """
    return [word_tokenize(text.lower()) for text in texts]

def train_lda_model(texts, num_topics=5):
    """
    Train an LDA topic model on the given texts.
    """
    processed_texts = preprocess_topic_texts(texts)
    dictionary = Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    return lda_model, dictionary, corpus

def get_topics(lda_model, top_n=10):
    """
    Get the topics from the trained LDA model.
    """
    return lda_model.print_topics(num_words=top_n)
