from sklearn.metrics.pairwise import cosine_similarity
import torch

def get_phobert_embedding(text, model, tokenizer, device):
    tokens = tokenizer(text, return_tensors="pt", padding=True, max_length=256)
    tokens = {key: value.to(device) for key, value in tokens.items()}  # Chuyển dữ liệu lên GPU nếu có
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state[:, 0, :].squeeze()

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
    vec1 = m1['embedding_film']
    vec2 = m2['embedding_film']
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
