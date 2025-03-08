import pandas as pd
import itertools
from tqdm import tqdm
import process_function as pf
import torch
from transformers import AutoModel, AutoTokenizer


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
     phobert_model = AutoModel.from_pretrained("vinai/phobert-base").to(device)
     tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

     df['embedding_film'] = df['describe'].apply(lambda x: pf.get_phobert_embedding(x, phobert_model, tokenizer, device))


     # Tạo cặp phim
     movie_pairs = list(itertools.combinations(df.index, 2))

     train_data = [pf.process_pair(df, pair) for pair in tqdm(movie_pairs, desc="Processing pairs")]

     # Lưu kết quả
     train_df = pd.DataFrame(train_data, columns=['Describe_1', 'Describe_2', 'Similarity_score'])
     train_df.to_csv("../data/data_similarity.csv", index=False)

     # Lưu embedding film
     embeddings = torch.stack(df['embedding_film'].tolist()) 
     torch.save(embeddings, "embedding_film.pt")
