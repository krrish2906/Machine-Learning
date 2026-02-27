import joblib
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recommend.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logging.info("🔁 Loading data...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    df = joblib.load(os.path.join(BASE_DIR, 'df_cleaned.pkl'))
    cosine_sim = joblib.load(os.path.join(BASE_DIR, 'cosine_sim.pkl'))
    logging.info("✅ Data loaded successfully.")
except Exception as e:
    logging.error("❌ Failed to load required files: %s", str(e))
    raise e

# Recommendation function:
def recommend_movies(movie_name, n_recommendations=5):
    logging.info("🎬 Recommending movies for: '%s'", movie_name)

    movie_index = df[df['title'].str.lower() == movie_name.lower()].index
    if len(movie_index) == 0:
        logging.warning("⚠️ Movie not found in dataset.")
        return None

    movie_index = movie_index[0]
    similarity_scores = list(enumerate(cosine_sim[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:n_recommendations + 1]

    movie_indices = [i[0] for i in similarity_scores]
    logging.info("✅ Top %d recommendations ready.", n_recommendations)

    # Create DataFrame with clean serial numbers starting from 1
    result_df = df[['title']].iloc[movie_indices].reset_index(drop=True)
    result_df.index = result_df.index + 1  # Start from 1 instead of 0
    result_df.index.name = "S.No."
    return result_df
