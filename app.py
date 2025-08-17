import streamlit as st
import re
import pickle

# ======== Lightweight text preprocessing (no NLTK) ========
# Minimal English stopword set (static, no downloads)
STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","any","both","each","few","more",
    "most","other","some","such","no","nor","not","only","own","same","so",
    "than","too","s","t","can","will","just","don","should","now"
}

def process_text(text: str) -> str:
    # remove urls, mentions, hashtags
    text = re.sub(r"(http\S+|www\.\S+)", " ", text)
    text = re.sub(r"[@#]\w+", " ", text)
    # keep letters and spaces
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    # normalize spaces + lowercase
    text = re.sub(r"\s+", " ", text).strip().lower()
    # simple tokenization + stopword filtering
    tokens = [tok for tok in text.split() if tok not in STOPWORDS and len(tok) > 1]
    return " ".join(tokens)

# ======== Load trained model + vectorizer ========
@st.cache_resource
def load_model():
    with open("sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

st.title("Sentiment Analysis App")
st.write("Enter any English comment and the model will analyze its sentiment.")

# Load
try:
    model, vectorizer = load_model()
except Exception as e:
    st.error("Model and vectorizer pickle files are required to run this app.")
    st.stop()

# ======== UI ========
user_comment = st.text_area("Enter your comment here:")

if st.button("Analyze Sentiment"):
    if not user_comment.strip():
        st.warning("Please enter a comment first.")
    else:
        cleaned = process_text(user_comment)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        sentiment = "Positive" if pred == 1 else "Negative"
        st.success(f"Predicted Sentiment: **{sentiment}**")
