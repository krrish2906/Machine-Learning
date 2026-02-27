import streamlit as st
from recommend import df, recommend_songs

# Set custom Streamlit page config
st.set_page_config(
    page_title="Music Recommender 🎵",
    page_icon="🎧",  # You can also use a path to a .ico or .png file
    layout="centered"
)

st.title("🎶 Instant Music Recommender")
song_list = sorted(df['song'].dropna().unique())
col1, col2 = st.columns(2)

with col1:
    selected_song = st.selectbox("🎵 Select a song:", song_list)
with col2:
    n_recommendations = st.selectbox("No. of songs:", [5,10,15])

if st.button("🚀 Recommend Similar Songs"):
    with st.spinner("Finding similar songs..."):
        recommendations = recommend_songs(selected_song, n_recommendations)
        if recommendations is None:
            st.warning("Sorry, song not found.")
        else:
            st.success("Top similar songs:")
            st.table(recommendations)
