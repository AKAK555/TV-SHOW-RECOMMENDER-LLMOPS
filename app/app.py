import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.pipeline import TVShowRecommendationPipeline
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Inject CSS for all text inputs
st.markdown("""
    <style>
    /* Target all text input fields */
    input[type="text"] {
        font-size: 18px !important;   /* Increase font size */
        height: 3em !important;       /* Make input box taller (optional) */
    }
    </style>
    """, unsafe_allow_html=True)


st.set_page_config(page_title="TV Show Recommender", page_icon=":clapper:", layout="wide")

@st.cache_resource
def get_pipeline():
    # Initialize pipeline with Chroma DB persistence directory
    pipeline = TVShowRecommendationPipeline(persist_directory="chroma_db")
    return pipeline

pipeline = get_pipeline()

st.title("TV Show Recommender System :clapper:")

query = st.text_input("Enter your TV Show preferences or description:", value="")

if query:
    with st.spinner("Generating recommendations for you..."):
        try:
            recommendation = pipeline.recommend(query)
        except Exception as e:
            st.error(f"Error generating recommendation: {e}")
        else:
            st.subheader("Recommended TV Shows:")
            st.write(recommendation)



