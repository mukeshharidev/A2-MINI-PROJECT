# grounded_qa.py
# pip install sentence-transformers rank-bm25 numpy requests
# ollama pull llama3.1:8b   (and keep `ollama serve` running)

import re
import numpy as np
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM = "llama3.1:8b"
MIN_SCORE = 0.35   # below this, the context is treated as not covering the question
TOP_K = 3

# --- 1. Your knowledge base (replace with chunks from your notes/PDFs) ---
DOCS = [
    {"src": "unit1.txt", "text": "A stack is a LIFO data structure. push adds an element on top, pop removes the top element."},
    {"src": "unit1.txt", "text": "A queue is a FIFO data structure. enqueue adds at the rear and dequeue removes from the front."},
    {"src": "unit2.txt", "text": "Binary search works on a sorted array and halves the search space each step, giving O(log n) time."},
    {"src": "unit2.txt", "text": "Bubble sort repeatedly swaps adjacent out-of-order elements and runs in O(n^2) in the worst case."},
]

# --- 2. Index ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_vecs = embedder.encode([d["text"] for d in DOCS], normalize_embeddings=True)
tokenize = lambda s: re.findall(r"\w+", s.lower())
bm25 = BM25Okapi([tokenize(d["text"]) for d in DOCS])


def minmax(x):
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min() + 1e-9)


def retrieve(query, k=TOP_K):
    q_vec = embedder.encode([query], normalize_embeddings=True)[0]
    dense = doc_vecs @ q_vec                       # cosine similarity
    sparse = bm25.get_scores(tokenize(query))
    hybrid = 0.6 * dense + 0.4 * minmax(sparse)    # simple score fusion
    idx = np.argsort(hybrid)[::-1][:k]
    return [(DOCS[i], float(dense[i])) for i in idx]


# --- 3. Grounded generation ---
def answer(query):
    hits = retrieve(query)
    if hits[0][1] < MIN_SCORE:
        return "I can't find this in the provided material, so I won't guess."

    context = "\n".join(f"[{i+1}] ({d['src']}) {d['text']}" for i, (d, _) in enumerate(hits))
    prompt = (
        "Answer ONLY using the context below. Cite sources like [1], [2]. "
        "If the context is insufficient, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    r = requests.post(OLLAMA_URL, json={
        "model": LLM, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.1},
    }, timeout=120)
    r.raise_for_status()
    return r.json()["response"].strip()


if __name__ == "__main__":
    while True:
        q = input("\nAsk (or 'q'): ").strip()
        if q.lower() == "q":
            break
        print(answer(q))
        s