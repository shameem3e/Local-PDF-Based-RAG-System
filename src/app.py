# app.py
import os
import argparse
from retriever import Retriever
from generator import Generator
import json

SRC_DIR = os.path.dirname(__file__)
PERSIST_DIR = os.path.join(SRC_DIR, "..", "persist")
METADATA_PATH = os.path.join(PERSIST_DIR, "metadata.json")

def interactive_chat(retriever: Retriever, generator: Generator, top_k=5):
    print("RAG interactive mode. Type 'exit' to quit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        hits = retriever.retrieve(q, top_k=top_k)
        # for demo we load snippets; in a production system, loader should store the full chunk text
        contexts = []
        # If you saved full chunk text in metadata (metadata[i]['full_text']), use that.
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            for h in hits:
                idx = h["index"]
                full = metadata[idx].get("full_text") or metadata[idx].get("text_snippet")
                contexts.append(full)
        else:
            contexts = [h["text"] for h in hits]

        answer = generator.generate_answer(q, contexts, max_new_tokens=120)
        print("\nAssistant:", answer)
        print("\nTop sources:")
        for h in hits:
            print(f" - {h['source']} (score: {h['score']:.3f})")

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("ingest", help="Ingest docs and build index (call src/ingest.py directly is also fine)")

    q = sub.add_parser("query", help="Make a single query")
    q.add_argument("--question", "-q", required=True)
    q.add_argument("--top_k", type=int, default=5)

    chat = sub.add_parser("chat", help="Interactive chat mode")
    chat.add_argument("--top_k", type=int, default=5)

    args = parser.parse_args()

    if args.cmd == "ingest":
        # call ingest script
        import subprocess, sys
        print("[app] Running ingest...")
        subprocess.check_call([sys.executable, os.path.join(SRC_DIR, "ingest.py")])
        return

    # for query and chat, load retriever and generator
    retr = Retriever()
    gen = Generator()

    if args.cmd == "query":
        hits = retr.retrieve(args.question, top_k=args.top_k)
        # build contexts
        contexts = []
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        for h in hits:
            idx = h["index"]
            contexts.append(metadata[idx].get("full_text") or metadata[idx].get("text_snippet"))
        ans = gen.generate_answer(args.question, contexts)
        print("Answer:\n", ans)
        print("\nTop sources:")
        for h in hits:
            print(f" - {h['source']} (score: {h['score']:.4f})")
    elif args.cmd == "chat":
        interactive_chat(retr, gen, top_k=args.top_k)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
