import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# ============ Text Preprocessing ============
def process_text(content):
    content = re.sub(r"http\S+", "", content)
    content = re.sub(r"@\w+", "", content)
    content = re.sub(r"#\w+", "", content)
    tokens = word_tokenize(content)
    tokens = [token for token in tokens if token.isalpha()]
    tokens = [token.lower() for token in tokens]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words("english")]
    porter_stemmer = PorterStemmer()
    tokens = [porter_stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

# ============ Load Trained Model & Vectorizer ============
@st.cache_resource
def load_model():
    model = pickle.load(open("sentiment_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

st.title("Sentiment Analysis App")
st.write("Enter any English comment and the model will analyze its sentiment.")

# Load model
try:
    model, vectorizer = load_model()
except:
    st.error("Model and vectorizer pickle files are required to run this app.")
    st.stop()

# ============ User Input ============
user_comment = st.text_area("Enter your comment here:")

if st.button("Analyze Sentiment"):
    if user_comment.strip() == "":
        st.warning("Please enter a comment first.")
    else:
        processed = process_text(user_comment)
        vector = vectorizer.transform([processed])
        prediction = model.predict(vector)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        st.success(f"Predicted Sentiment: **{sentiment}**")
