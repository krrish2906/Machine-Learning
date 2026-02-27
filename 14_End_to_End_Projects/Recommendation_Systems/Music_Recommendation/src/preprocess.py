import pandas as pd
import os
import re
import nltk
import joblib
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logging.info("🚀 Starting preprocessing...")

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "spotify_millsongdata.csv")

try:
    df = pd.read_csv(CSV_PATH).sample(10000)
    logging.info("✅ Dataset loaded in Dataframe & sampled: %d rows", len(df))
except Exception as e:
    logging.error("❌ Failed to load dataset: %s", str(e))
    raise e

# Drop column 'link':
df = df.drop(columns='link', axis=1, errors='ignore').reset_index(drop=True)
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    cleaned_text = " ".join(tokens)
    return cleaned_text

logging.info("🧹 Cleaning text...")
df['processed_text'] = df['text'].apply(preprocess_text)
logging.info("✅ Text cleaned.")


# Vectorization:
logging.info("🔠 Vectorizing using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df['processed_text'])
logging.info("✅ TF-IDF matrix shape: %s", tfidf_matrix.shape)


# Cosine Similarity
logging.info("📐 Calculating cosine similarity...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
logging.info("✅ Cosine similarity matrix generated.")

joblib.dump(df, 'df_cleaned.pkl')
joblib.dump(tfidf_matrix, 'tfidf_matrix.pkl')
joblib.dump(cosine_sim, 'cosine_sim.pkl')
logging.info("💾 Data saved to disk.")

logging.info("✅ Preprocessing complete.")