from flask import Flask, render_template_string, request
from search_engine import preprocess, build_inverted_index, get_doc_vectors, cosine_similarity
import os

app = Flask(__name__)

# 一次性加载文档、索引、向量（和main.py逻辑完全一致）
folder = "test_docs"
docs = {}
if os.path.exists(folder):
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            full_path = os.path.join(folder, filename)
            with open(full_path, "r", encoding="utf-8") as f:
                text = f.read()
                words = preprocess(text)
                docs[filename] = words

index = build_inverted_index(docs)
doc_vecs = get_doc_vectors(docs, index)

# 简单网页模板，内嵌在代码里，不用额外html文件
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Mini Search Engine</title>
    <style>
        body{font-family: -apple-system, sans-serif;max-width:700px;margin:40px auto;padding:0 20px;}
        .box{margin:20px 0;}
        input[type="text"]{width:100%;padding:10px;font-size:16px;}
        button{padding:10px 20px;font-size:16px;cursor:pointer;}
        .result{margin-top:20px;padding:15px;border:1px solid #ddd;border-radius:8px;}
    </style>
</head>
<body>
    <h1>Simple Full Text Search</h1>
    <form method="GET">
        <div class="box">
            <input type="text" name="q" placeholder="Input search keyword..." value="{{query}}">
        </div>
        <button type="submit">Search</button>
    </form>
    {% if results %}
    <div class="result">
        <h3>Search Results</h3>
        <ul>
        {% for name,score in results %}
            <li>{{name}} | similarity: {{ "%.4f"|format(score) }}</li>
        {% endfor %}
        </ul>
    </div>
    {% endif %}
</body>
</html>
"""

@app.route("/")
def home():
    query = request.args.get("q","").strip()
    results = []
    if query:
        query_words = preprocess(query)
        query_vec = {}
        for w in query_words:
            query_vec[w] = query_vec.get(w,0)+1
        for doc_name, vec in doc_vecs.items():
            score = cosine_similarity(query_vec, vec)
            if score>0:
                results.append((doc_name, score))
        results.sort(key=lambda x:x[1], reverse=True)
    return render_template_string(HTML_TEMPLATE, query=query, results=results)

if __name__ == "__main__":
    app.run(debug=True)
