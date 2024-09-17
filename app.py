import streamlit as st
import google.generativeai as genai
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import FER as face

st.set_page_config(
    page_title="Movie Chatbot with FER",
    layout="centered",
    initial_sidebar_state="auto",
)

@st.cache_resource
def download_nltk():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

download_nltk()


st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Go to", ["Movie Chatbot", "Facial Expression Recognition"])

api_key = "AIzaSyCYw0zlKdcUZ_yDzVG3ne7sFY6a5VFsF5g"
genai.configure(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

def remove_year_from_title(title):
    return re.sub(r'\(\d{4}\)', '', title).strip()

@st.cache_data
def load_and_lemmatize_keywords():
    lemmatizer = WordNetLemmatizer()
    movie_keywords = [
        'movie', 'film', 'actor', 'actress', 'director', 'release',
        'trailer', 'cinema', 'genre', 'plot', 'character', 'rating',
        'box office', 'award', 'screenplay', 'sequel', 'franchise',
        'animation', 'documentary', 'comedy', 'drama', 'horror',
        'thriller', 'romance', 'sci-fi', 'fantasy', 'action', 'cast'
    ]


    titles = pd.read_csv('./ml-32m/movies.csv')
    titles['title'] = titles['title'].apply(remove_year_from_title).str.lower()
    title_list = titles['title'].tolist()

    lemmatized_keywords = set()
    for keyword in movie_keywords:
        tokens = word_tokenize(keyword)
        lemmas = [lemmatizer.lemmatize(token) for token in tokens]
        lemmatized_keywords.update(lemmas)
    for title in title_list:
        tokens = word_tokenize(title)
        lemmas = [lemmatizer.lemmatize(token) for token in tokens]
        lemmatized_keywords.update(lemmas)

    return lemmatized_keywords

lemmatized_keywords = load_and_lemmatize_keywords()

def is_movie_related(query):
    if not lemmatized_keywords:
        return False

    lemmatizer = WordNetLemmatizer()
    query_tokens = word_tokenize(query.lower())
    query_lemmas = set(lemmatizer.lemmatize(token) for token in query_tokens)

    return not query_lemmas.isdisjoint(lemmatized_keywords)

def generate_gemini_response(prompt):
    try:
        with st.spinner('Generating response...'):
            model = genai.GenerativeModel("gemini-1.5-flash-latest")
            response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I encountered an error while processing your request."

def movie_chatbot():
    st.title("🎬 Movie Chatbot")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything about movies..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        if is_movie_related(prompt):
            with st.chat_message("assistant"):
                assistant_reply = generate_gemini_response(prompt)
                st.markdown(assistant_reply)

            st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
        else:
            with st.chat_message("assistant"):
                assistant_reply = "I can only help with movie-related queries. Please ask about movies!"
                st.markdown(assistant_reply)

            st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

def facial_expression_recognition():
    st.title("📸 Facial Expression Recognition")

    try:
        face.Camera()
        st.write("Use your camera to detect facial expressions in real-time.")
    except Exception as e:
        st.error(f"Error accessing the camera: {e}")
        st.write("Please ensure that your camera is connected and accessible.")

if app_mode == "Movie Chatbot":
    movie_chatbot()
elif app_mode == "Facial Expression Recognition":
    facial_expression_recognition()