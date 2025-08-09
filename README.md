# 📚 Local-PDF-Based RAG System

A **Retrieval-Augmented Generation (RAG)** system that processes **local PDF documents** to answer user queries in a context-aware manner — without relying on ChatGPT or paid APIs.  
Simply place your documents in the `docs/` folder, run the ingestion process, and start chatting from the terminal.

---

## 📌 Features
- 📄 **PDF Document Support** — Works with `.pdf`, `.txt`, `.md` files.
- 🔍 **Semantic Search** — Uses embeddings + vector database for relevant chunk retrieval.
- 🧠 **Local LLM** — Runs on pre-trained Hugging Face models (no internet needed after setup).
- 💬 **Interactive Chat Mode** — Terminal-based chatbot for follow-up questions.
- ⚡ **Fast Retrieval** — Powered by FAISS for quick vector search.
- 🛠 **Customizable** — Change chunk size, overlap, or switch to larger models easily.

---

## 📂 Folder Structure
my_rag_project/

├─ docs/ # Place your PDF/TXT/MD files here

├─ persist/ # Saved embeddings, index, and metadata

├─ src/

│ ├─ utils.py # PDF text extraction & chunking

│ ├─ ingest.py # Embeds docs & stores in FAISS

│ ├─ retriever.py # Retrieves top-matching chunks

│ ├─ generator.py # Generates answer using local LLM

│ └─ app.py # Main entry: query or chat mode

├─ requirements.txt # Python dependencies

└─ README.md # Project documentation


---

## ⚙️ Setup Guide

### 1️⃣ Clone Repository
```bash
git clone https://github.com/shameem3e/Local-PDF-Based-RAG-System.git
cd Local-PDF-Based-RAG-System

```
### **2️⃣ Create Virtual Environment**
```bash
python -m venv .venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

```
### **3️⃣ Install Requirements**
```bash
pip install -r requirements.txt

```
### **4️⃣ Add Documents**
Place your `.pdf`, `.txt`, or `.md` files into the `docs/` folder.

### **5️⃣ Ingest Documents**
```bash
python src/ingest.py

```
### **6️⃣ Query or Chat**
* Single Question
```bash
python src/app.py query --question "YOUR QUESTION?"

```
* Chat Mode
```bash
python src/app.py chat

```
Type your question from the PDF and chat with your RAG system.

## 📜 Code Overview
| File             | Description                                                                |
| ---------------- | -------------------------------------------------------------------------- |
| **utils.py**     | Extracts and chunks text from PDF/TXT/MD files.                            |
| **ingest.py**    | Converts chunks into embeddings and saves them into FAISS index.           |
| **retriever.py** | Finds most relevant text chunks based on the query.                        |
| **generator.py** | Uses a Hugging Face model to generate final answer from retrieved context. |
| **app.py**       | CLI tool for ingestion, querying, and interactive chatting.                |

## ❓ FAQ
Q: Do I need the internet to run it?

A: Only for the first model download. After that, it works offline.

Q: Can I use other file types?

A: Yes, `.txt` and `.md` work. For others, add parsing logic in `utils.py`.

Q: It’s slow, what can I do?

A: Use a smaller model (`flan-t5-small`) or run on GPU.

Q: FAISS not installing?

A: Use `pip install faiss-cpu` or switch to `scikit-learn` fallback.

## 🛠 Tech Stack
* Python 3.9+
* FAISS — Vector database for fast similarity search
* Sentence-Transformers — Embedding generation
* Hugging Face Transformers — Local language models
* PyPDF2 / pdfplumber — PDF text extraction

## 🚀 Future Improvements
* Add Google Search as optional data source
* Support multimodal RAG (PDF + Images)
* Add web UI instead of terminal
* Improve answer summarization

## 👨‍💻 Author
[MD. Shameem Ahammed](https://sites.google.com/view/shameem3e)

Graduate Student | AI & ML Enthusiast

---
