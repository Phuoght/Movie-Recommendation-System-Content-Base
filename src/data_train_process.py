import pandas as pd
import itertools
from tqdm import tqdm
import process_function as pf
import torch
from transformers import AutoModel, AutoTokenizer
import pickle
import numpy as np

if __name__ == "__main__":
     # Load data
     df = pd.read_csv('../data/data-film-final.csv')

    # Chuẩn hóa dữ liệu
     df['title'] = df['title'].str.lower()
     df['genre'] = df['genre'].str.lower()
     df['director'] = df['director'].str.lower()
     df['actor'] = df['actor'].astype(str).str.split(', ').apply(lambda x: set(x))

     # Load Phobert
     device = "cuda" if torch.cuda.is_available() else "cpu"
     phobert_model = AutoModel.from_pretrained("vinai/phobert-base")
     tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
     embeddings = pf.extract_embeddings(df['describe'].astype(str).tolist(), tokenizer, phobert_model, max_length=256, stride=128, device = device, batch_size=32)
     df['embedding_film'] = embeddings
     
     # Lưu vào file .pkl
     with open('../data/embedding_train.pkl', 'wb') as f:
          pickle.dump([emb.cpu().numpy() for emb in embeddings], f)

     # Tạo cặp phim
     movie_pairs = list(itertools.combinations(df.index, 2))

     train_data = [pf.process_pair(df, pair) for pair in tqdm(movie_pairs, desc="Processing pairs")]

     # Lưu kết quả
     train_df = pd.DataFrame(train_data, columns=['Describe_1', 'Describe_2', 'Similarity_score'])
     train_df.to_csv("../data/data_similarity.csv", index=False)