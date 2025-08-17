import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
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

# ============ Load Dataset ============
# هنا انت ممكن تحط الـ Sentiment140 dataset
# للتجربة هنعمل DataFrame بسيط يدوي
data = pd.DataFrame({
    "text": [
        "I love this product, it is amazing!",
        "This is the worst thing I ever bought.",
        "Absolutely fantastic experience.",
        "I hate it so much.",
        "Really good quality and service.",
        "Terrible, I want my money back."
    ],
    "target": [1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative
})

# Preprocess
data["text"] = data["text"].apply(process_text)

X = data["text"].values
y = data["target"].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save Model & Vectorizer
pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model and vectorizer have been saved as .pkl files")
