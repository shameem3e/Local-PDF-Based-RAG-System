# utils.py
import os
from PyPDF2 import PdfReader

def extract_text_from_file(path: str) -> str:
    """
    Support .pdf, .txt, .md
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        try:
            reader = PdfReader(path)
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(pages)
        except Exception as e:
            print(f"[utils] Error reading PDF {path}: {e}")
            return ""
    else:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            print(f"[utils] Error reading text file {path}: {e}")
            return ""

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """
    Simple character-based chunking with overlap.
    Returns list of chunks (strings).
    """
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
