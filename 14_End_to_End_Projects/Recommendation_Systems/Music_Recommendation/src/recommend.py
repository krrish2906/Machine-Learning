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
def recommend_songs(song_name, num_recommendations=5):
    logging.info("🎵 Recommending songs for: '%s'", song_name)
    song_index = df[df['song'].str.lower() == song_name.lower()].index

    if len(song_index) == 0:
        logging.warning("⚠️ Song not found in dataset.")
        return None

    song_index = song_index[0]

    similarity_scores = list(enumerate(cosine_sim[song_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:num_recommendations+1]

    song_indices = [i[0] for i in similarity_scores]
    logging.info("✅ Top %d recommendations ready.", num_recommendations)

    # Create DataFrame with clean serial numbers starting from 1:
    result_df = df[['artist', 'song']].iloc[song_indices].reset_index(drop=True)
    result_df.index = result_df.index + 1  # Start from 1 instead of 0
    result_df.index.name = "S.No."
    return result_df
