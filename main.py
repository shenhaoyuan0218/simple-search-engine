import os
from search_engine import preprocess, build_inverted_index, get_doc_vectors, cosine_similarity

def load_documents(folder_path):
    """Read all txt files from target folder"""
    docs = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            full_path = os.path.join(folder_path, filename)
            with open(full_path, "r", encoding="utf-8") as f:
                text = f.read()
                words = preprocess(text)
                docs[filename] = words
    return docs

def main():
    folder = "test_docs"
    if not os.path.exists(folder):
        os.mkdir(folder)
        print(f"Created folder {folder}, please put your .txt files inside it.")
        return

    print("Loading documents and building index...")
    documents = load_documents(folder)
    if len(documents) == 0:
        print("No txt files found in test_docs folder!")
        return

    index = build_inverted_index(documents)
    doc_vecs = get_doc_vectors(documents, index)
    total_docs = len(documents)
    print(f"Loaded {total_docs} documents, ready for search.")
    print("Input your query, type 'exit' to quit.\n")

    while True:
        query = input("> ")
        if query.strip().lower() == "exit":
            break
        query_words = preprocess(query)
        query_vec = {}
        for w in query_words:
            query_vec[w] = query_vec.get(w,0)+1

        results = []
        for doc_name, vec in doc_vecs.items():
            score = cosine_similarity(query_vec, vec)
            if score > 0:
                results.append((doc_name, score))
        # sort from high similarity to low
        results.sort(key=lambda x:x[1], reverse=True)
        print("\n==== Search Result ====")
        for name, score in results:
            print(f"{name} | similarity: {score:.4f}")
        print()

if __name__ == "__main__":
    main()
