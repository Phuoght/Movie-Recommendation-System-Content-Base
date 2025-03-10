from sklearn.metrics.pairwise import cosine_similarity
import torch
from tqdm import tqdm

def split_into_chunks(tokens, max_length=256, stride=128):
    chunks = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i + max_length]
        if len(chunk) > 0:
            chunks.append(chunk)
    return chunks

# Hàm trích xuất embedding tối ưu
def extract_embeddings(texts, tokenizer, model, max_length=256, stride=128, device = 'gpu', batch_size=32):
    embeddings = []
    model.to(device)
    model.eval()

    for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
        batch_texts = texts[i:i + batch_size]
        batch_chunks = []
        for text in batch_texts:
            text = str(text).strip()
            tokens = tokenizer.encode(text, add_special_tokens=True)
            chunks = split_into_chunks(tokens, max_length, stride)
            if not chunks:
                embeddings.append(torch.zeros(768).to(device))
                continue
            batch_chunks.append(chunks)

        # Chuẩn bị input cho batch
        max_chunks = max(len(chunks) for chunks in batch_chunks)
        input_ids_batch = []
        attention_mask_batch = []
        
        for chunks in batch_chunks:
            for chunk in chunks:
                padded_chunk = chunk + [tokenizer.pad_token_id] * (max_length - len(chunk))
                attention_mask = [1] * len(chunk) + [0] * (max_length - len(chunk))
                input_ids_batch.append(padded_chunk)
                attention_mask_batch.append(attention_mask)
            # Padding thêm nếu số chunk nhỏ hơn max_chunks
            for _ in range(max_chunks - len(chunks)):
                input_ids_batch.append([tokenizer.pad_token_id] * max_length)
                attention_mask_batch.append([0] * max_length)

        # Chuyển thành tensor
        input_ids = torch.tensor(input_ids_batch).to(device)
        attention_mask = torch.tensor(attention_mask_batch).to(device)

        # Trích xuất embedding
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # Tổng hợp embedding cho từng văn bản
        start_idx = 0
        for chunks in batch_chunks:
            num_chunks = len(chunks)
            if num_chunks > 0:
                text_emb = cls_embeddings[start_idx:start_idx + num_chunks].mean(dim=0)
            else:
                text_emb = torch.zeros(768).to(device)
            embeddings.append(text_emb)
            start_idx += max_chunks

    return embeddings

def similar_title(m1, m2):
    title_words1 = m1['title'].split()
    title_words2 = m2['title'].split()
    word_t1 = ''
    word_t2 = ''
    for word in title_words1:
        if ':' in word:
            word_t1 += word
            break
        else:
            word_t1 += word
    for word in title_words2:
        if ':' in word:
            word_t2 += word
            break
        else:
            word_t2 += word
    if word_t1 == word_t2:  # xử lý loạt phim
        return 1
    return len(set(title_words1) & set(title_words2)) / max(len(set(title_words1)), len(set(title_words2)))

# Hàm tính độ tương đồng của đạo diễn
def similar_director(m1, m2):
    return 1 if m1['director'] == m2['director'] else 0

# Hàm tính độ tương đồng thể loại
def similar_genre(m1, m2):
    return 1 if m1['genre'] == m2['genre'] else 0

# Hàm tính độ tương đồng diễn viên
def similar_actor(m1, m2):
    max_len_movie = max(len(m1['actor']), len(m2['actor']))
    if max_len_movie == 0:
        return 0
    return len(m1['actor'] & m2['actor']) / max_len_movie

# Hàm tính độ tương đồng mô tả (chuyển sang PyTorch)
def similar_describe(m1, m2):
    vec1 = torch.tensor(m1['embedding_film']).unsqueeze(0)
    vec2 = torch.tensor(m2['embedding_film']).unsqueeze(0)
    similarity = torch.nn.functional.cosine_similarity(vec1, vec2).item()
    return similarity

# Hàm tính toán điểm tương đồng giữa 2 phim
def process_pair(df, pair):
    i, j = pair
    movie1, movie2 = df.iloc[i], df.iloc[j]

    # Trọng số
    w_title = 0.2
    w_describe = 0.4
    w_genre = 0.15
    w_director = 0.15
    w_actor = 0.1

    title_score = similar_title(movie1, movie2)
    director_score = similar_director(movie1, movie2)
    genre_score = similar_genre(movie1, movie2)
    actor_score = similar_actor(movie1, movie2)
    describe_score = similar_describe(movie1, movie2)

    # Tính điểm tổng
    similarity_score = (
        w_title * title_score + w_describe * describe_score +
        w_genre * genre_score + w_director * director_score +
        w_actor * actor_score
    )

    return [movie1['describe'], movie2['describe'], round(similarity_score, 2)]
