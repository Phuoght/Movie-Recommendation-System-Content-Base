import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import streamlit.components.v1 as components


st.set_page_config(page_title="Movie Recommendation ", page_icon='🎞️')

st.markdown(
    """
    <style>

    /* Đặt nền full màn hình */
    .stApp {
        background-image: url('https://t3.ftcdn.net/jpg/05/13/16/14/360_F_513161494_SVB7FFjAufL3VFfFV75HvLcGdBVAJqru.jpg');
        background-size: cover;
        background-position: center;
    }

    /* Tăng kích thước hiển thị */
    .block-container {
        padding: 1rem 2rem;
        max-width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Danh sách phim
movies_df = pd.read_csv('data/data-film-final.csv')

movie_head = movies_df.head(30)

# Danh sach describe phim
movie_des = movies_df
movie_des = movie_des.set_index(movies_df['title'])
movie_des = movie_des['describe']

st.markdown("""
    <h1 style="text-align: center; color: #D21F58FF; margin-bottom:5px; margin-top:20px" > 
            Movie Recommendation System
    </h1>
            """, unsafe_allow_html=True)

def show_movie(df):
    html_content = """
        <style>
            .movie-container {
                display: flex;
                overflow-x: auto;
                padding: 10px;
                white-space: nowrap;
                scrollbar-width: thin;
                scrollbar-color: #571B1BFF #6E1515FF;
            }
            .movie-container::-webkit-scrollbar {
                height: 3px;
            }
            .movie-container::-webkit-scrollbar-thumb {
                background: #888 #9B1111FF;
                border-radius: 10px;
            }
            .movie-item {
                position: relative;
                margin-right: 15px;
                border-radius: 10px;
                flex: 0 0 auto;
                text-align: center;
            }
            .movie-item img {
                width: 290px;
                height: 150px;
                border-radius: 10px;
                object-fit: cover;
            }
            .movie-title {
                position: absolute;
                bottom: 0;
                width: 100%;
                background: #201722A6;
                color: white;
                text-align: center;
                font-size: 14px;
            }
        </style>
        <div class="movie-container">
    """

    # Thêm từng phim vào HTML
    for _, row in df.iterrows():
        html_content += f"""
            <div class="movie-item">
                <a href="{row['url_film']}" target="_blank">
                    <img src="{row['img_film']}" alt="{row['title']}">
                    <div class="movie-title">{row['title']}</div>
                </a>
            </div>
        """

    html_content += "</div>"

    # Hiển thị HTML bằng components.v1.html
    components.html(html_content, height=200)

# Display initial 30 movies
show_movie(movie_head)


# Search input
search_query = st.text_input("Search:", "", key="search")

# Load SBERT fine tuned
model = SentenceTransformer("fine_tuned_sbert_movies")

# Load embedding movies
movies_embedding = torch.load("movies_embedding.pt")  # Tải embedding từ file

def recommend_movies(query, top_k=10):
    query_des = movie_des.loc[query]
    query_embedding = model.encode(query_des, convert_to_tensor=True)# Encode query
    similarity_scores = torch.nn.functional.cosine_similarity(query_embedding, movies_embedding)  # Tính cosine similarity

    top_results = torch.argsort(similarity_scores, descending=True)[:top_k]
    recommended_movies = movies_df.iloc[top_results.numpy()]  # Lấy thông tin phim
    return recommended_movies

if search_query:
    st.markdown("<h3 style='color: #F1799FFF;'>Recommend for you:</h3>", unsafe_allow_html=True)

    # Dự đoán top 10 phim tương tự
    recommended_movies = recommend_movies(search_query)

    # Hiển thị danh sách phim gợi ý
    show_movie(recommended_movies)
