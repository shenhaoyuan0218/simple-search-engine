## 2. search_engine.py
import string
import math

# English stop words
stopwords = {"the", "a", "an", "of", "is", "in", "and", "or", "to", "for", "be"}

def preprocess(text: str):
    """
    Text preprocessing: lowercase, remove punctuation, split words, filter stopwords
    """
    text = text.lower()
    # delete all punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    words = text.split()
    words = [w for w in words if w not in stopwords]
    return words

def build_inverted_index(documents):
    """
    Build inverted index: word -> list[(doc_name, word_count)]
    documents: dict, key: filename, value: list of words
    """
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

def compute_tf(word, doc_words):
    """TF: term frequency"""
    return doc_words.count(word) / len(doc_words)

def compute_idf(word, index, total_docs):
    """IDF: inverse document frequency"""
    doc_contained = len(index.get(word, []))
    return math.log(total_docs / (1 + doc_contained))

def get_doc_vectors(documents, index):
    """Generate TF-IDF vector for every document"""
    total_docs = len(documents)
    doc_vecs = {}
    for doc_name, words in documents.items():
        vec = {}
        unique_words = set(words)
        for w in unique_words:
            tf = compute_tf(w, words)
            idf = compute_idf(w, index, total_docs)
            vec[w] = tf * idf
        doc_vecs[doc_name] = vec
    return doc_vecs

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two sparse vectors"""
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
