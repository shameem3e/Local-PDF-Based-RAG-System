# generator.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Model choices:
# - Small CPU-friendly: "google/flan-t5-small"
# - Bigger/better: "google/flan-t5-base" or "google/flan-t5-large" (need more RAM/GPU)
DEFAULT_GEN_MODEL = "google/flan-t5-small"

class Generator:
    def __init__(self, model_name: str = DEFAULT_GEN_MODEL, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"[generator] Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        # model.config.max_length may be available; tokenizer.model_max_length as fallback
        self.max_input_tokens = min( self.tokenizer.model_max_length, 1024 )

    def generate_answer(self, question: str, contexts: list, max_new_tokens: int = 150):
        """
        contexts: list of strings (retrieved chunks) — concatenate them into a context block
        """
        # join contexts (keep it short to fit token limit)
        context_text = "\n\n---\n\n".join(contexts)
        prompt = (
            "You are a helpful assistant. Use the context below to answer the question. "
            "If the answer is not contained in the context, say 'I don't know'.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
        )
        # tokenize (truncate if needed)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=self.max_input_tokens).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4)
        answer = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return answer
