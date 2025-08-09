# ingest.py
import os
import json
import glob
from utils import extract_text_from_file, chunk_text, ensure_dir
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm

# Try to import faiss; if not available we will fallback to storing embeddings as numpy array
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "persist")
INDEX_PATH = os.path.join(PERSIST_DIR, "faiss.index")
EMBEDINGS_NPY = os.path.join(PERSIST_DIR, "embeddings.npy")
METADATA_PATH = os.path.join(PERSIST_DIR, "metadata.json")

def collect_chunks_from_docs(docs_dir: str, chunk_size=1000, overlap=200):
    allowed = [".pdf", ".txt", ".md"]
    chunks = []
    metadata = []
    file_paths = []
    for ext in allowed:
        file_paths.extend(glob.glob(os.path.join(docs_dir, f"*{ext}")))
    file_paths = sorted(file_paths)
    for fp in file_paths:
        print(f"[ingest] reading {fp}")
        text = extract_text_from_file(fp)
        file_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for i, ch in enumerate(file_chunks):
            chunks.append(ch)
            metadata.append({
                "source": os.path.basename(fp),
                "chunk_index": i,
                "text_snippet": ch[:200]  # short preview
            })
    return chunks, metadata

def build_index(chunks, metadata, embed_model_name="all-MiniLM-L6-v2"):
    ensure_dir(PERSIST_DIR)
    print(f"[ingest] Loading embedder: {embed_model_name}")
    embedder = SentenceTransformer(embed_model_name)
    # encode
    print(f"[ingest] Encoding {len(chunks)} chunks ...")
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    # normalize for cosine similarity with faiss.IndexFlatIP
    embeddings = normalize(embeddings, axis=1).astype("float32")
    # Save metadata
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    if FAISS_AVAILABLE:
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        faiss.write_index(index, INDEX_PATH)
        print(f"[ingest] FAISS index saved to {INDEX_PATH}")
    else:
        # fallback: save numpy embeddings and metadata
        np.save(EMBEDINGS_NPY, embeddings)
        print(f"[ingest] FAISS not available. Embeddings saved to {EMBEDINGS_NPY}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest docs & build vector index")
    parser.add_argument("--docs_dir", default=os.path.join(os.path.dirname(__file__), "..", "docs"),
                        help="Folder containing documents (.pdf/.txt/.md)")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--embed_model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    chunks, metadata = collect_chunks_from_docs(args.docs_dir, args.chunk_size, args.overlap)
    if not chunks:
        print("[ingest] No chunks found. Put your files in the docs/ folder.")
        return
    build_index(chunks, metadata, embed_model_name=args.embed_model)
    print("[ingest] Done.")

if __name__ == "__main__":
    main()
