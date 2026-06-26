from sentence_transformers import SentenceTransformer

model = SentenceTransformer("/home/jncsnlp4/tb/Qwen2-VL-main/sentence-transformers/all-MiniLM-L6-v2")

def text_sim(sentence1,sentence2):

    # Compute embeddings for both lists
    embeddings1 = model.encode(sentence1)
    embeddings2 = model.encode(sentence2)

    # Compute cosine similarities
    similarities = model.similarity(embeddings1, embeddings2)

    # print(similarities.item())
    return similarities.item()
# Output the pairs with their score
# for idx_i, sentence1 in enumerate(sentences1):
#     print(sentence1)
#     for idx_j, sentence2 in enumerate(sentences2):
#         print(f" - {sentence2: <30}: {similarities[idx_i][idx_j]:.4f}")