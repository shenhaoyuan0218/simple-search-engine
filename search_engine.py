import string
import math

stopwords = {"the", "a", "an", "of", "is", "in", "and", "or", "to", "for", "be"}

def preprocess(text: str):
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    words = text.split()
    words = [w for w in words if w not in stopwords]
    return words

def build_inverted_index(documents):
    index = {}
    for doc_id, words in documents.items():
        word_count = {}
        for w in words:
            word_count[w] = word_count.get(w, 0) + 1
        for word, cnt in word_count.items():
            if word not in index:
                index[word] = []
            index[word].append((doc_id, cnt))
    return index
def get_doc_vectors(documents, index):
    doc_vecs = {}
    for doc_name, words in documents.items():
        vec = {}
        for w in words:
            vec[w] = vec.get(w,0)+1
        doc_vecs[doc_name] = vec
    return doc_vecs

def cosine_similarity(vec1, vec2):
    dot_product = 0
    mag1 = 0
    mag2 = 0
    for key in vec1:
        dot_product += vec1[key] * vec2.get(key, 0)
        mag1 += vec1[key] ** 2
    for key in vec2:
        mag2 += vec2[key] ** 2
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot_product / (math.sqrt(mag1) * math.sqrt(mag2))
