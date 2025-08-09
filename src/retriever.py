# retriever.py
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False
    from sklearn.neighbors import NearestNeighbors

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "persist")
INDEX_PATH = os.path.join(PERSIST_DIR, "faiss.index")
EMBEDINGS_NPY = os.path.join(PERSIST_DIR, "embeddings.npy")
METADATA_PATH = os.path.join(PERSIST_DIR, "metadata.json")

EMBED_MODEL = "all-MiniLM-L6-v2"

class Retriever:
    def __init__(self, embed_model_name=EMBED_MODEL):
        self.embedder = SentenceTransformer(embed_model_name)
        # load metadata
        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(f"Metadata not found at {METADATA_PATH}. Run ingest first.")
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        if FAISS_AVAILABLE:
            if not os.path.exists(INDEX_PATH):
                raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}. Run ingest first.")
            self.index = faiss.read_index(INDEX_PATH)
            self.use_faiss = True
        else:
            if not os.path.exists(EMBEDINGS_NPY):
                raise FileNotFoundError(f"Embeddings file not found at {EMBEDINGS_NPY}. Run ingest first.")
            self.embeddings = np.load(EMBEDINGS_NPY)
            # Prebuild a NearestNeighbors object for fast queries
            self.nn = NearestNeighbors(n_neighbors=10, metric='cosine').fit(self.embeddings)
            self.use_faiss = False

    def retrieve(self, query: str, top_k: int = 5):
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        q_emb = normalize(q_emb, axis=1).astype("float32")
        results = []
        if self.use_faiss:
            D, I = self.index.search(q_emb, top_k)  # D: scores, I: indices
            for score, idx in zip(D[0], I[0]):
                if idx < 0:
                    continue
                meta = self.metadata[idx]
                results.append({
                    "score": float(score),
                    "index": int(idx),
                    "source": meta.get("source"),
                    "chunk_index": meta.get("chunk_index"),
                    "text": meta.get("text_snippet")  # short snippet; full chunk stored in ingestion if needed
                })
        else:
            distances, indices = self.nn.kneighbors(q_emb, n_neighbors=top_k)
            # sklearn NearestNeighbors uses distance; smaller = more similar
            for dist, idx in zip(distances[0], indices[0]):
                meta = self.metadata[int(idx)]
                results.append({
                    "score": float(1 - dist),  # rough similarity score
                    "index": int(idx),
                    "source": meta.get("source"),
                    "chunk_index": meta.get("chunk_index"),
                    "text": meta.get("text_snippet")
                })
        return results

if __name__ == "__main__":
    # quick test
    r = Retriever()
    q = "What are the safety procedures?"
    res = r.retrieve(q, top_k=3)
    print(res)
