import pandas as pd
from sentence_transformers import SentenceTransformer
import itertools
from tqdm import tqdm
import process_function as pf

if __name__ == "__main__":
     # Load data
     df = pd.read_csv('../data/data-film-final.csv')

    # Chuẩn hóa dữ liệu
     df['title'] = df['title'].str.lower()
     df['genre'] = df['genre'].str.lower()
     df['director'] = df['director'].str.lower()
     df['actor'] = df['actor'].astype(str).str.split(', ').apply(lambda x: set(x))

     # Tải SentenceTransformer lên GPU
     sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
     df['embedding_film'] = df['describe'].apply(lambda x: sbert_model.encode(x))

     # Tạo cặp phim
     movie_pairs = list(itertools.combinations(df.index, 2))

     train_data = [pf.process_pair(df, pair) for pair in tqdm(movie_pairs, desc="Processing pairs")]

     # Lưu kết quả
     train_df = pd.DataFrame(train_data, columns=['Movie_1', 'Movie_2', 'Describe_1', 'Describe_2', 'Similarity_score'])
     train_df.to_csv("../data/data_similarity.csv", index=False)
