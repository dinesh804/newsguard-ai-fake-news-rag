from __future__ import annotations
import os, re, json, time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

import requests
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

LABELS = {0: "REAL", 1: "FAKE"}

@dataclass
class EvidenceChunk:
    source: str
    text: str
    chunk_id: int

@dataclass
class NewsPrediction:
    label: str
    confidence: Optional[float]
    model_name: str
    note: str
    inference_ms: Optional[float] = None

class NewsClassifierAgent:
    """Loads a locally fine-tuned RoBERTa model if one exists; never fabricates predictions."""
    def __init__(self, model_dir: str = "models/roberta-fake-news"):
        self.model_dir = model_dir
        self.tokenizer = None
        self.model = None
        self.load_error = None
        if os.path.isdir(model_dir) and os.path.isfile(os.path.join(model_dir, "config.json")):
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
                self.model.eval()
                self.torch = torch
            except Exception as exc:
                self.load_error = str(exc)

    @property
    def ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def predict(self, text: str) -> NewsPrediction:
        text = (text or "").strip()
        if not text:
            raise ValueError("Enter news text first.")
        if not self.ready:
            return NewsPrediction("MODEL_NOT_TRAINED", None, "RoBERTa", "Train the model with train_roberta.py before requesting a classification.")
        started = time.perf_counter()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
            probs = self.torch.softmax(logits, dim=-1)[0].cpu().tolist()
        idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        id2label = getattr(self.model.config, "id2label", {}) or {}
        raw_label = str(id2label.get(idx, LABELS.get(idx, f"CLASS_{idx}"))).upper()
        if "FAKE" in raw_label or "FALSE" in raw_label:
            label = "FAKE"
        elif "REAL" in raw_label or "TRUE" in raw_label or "GENUINE" in raw_label:
            label = "REAL"
        else:
            label = raw_label
        return NewsPrediction(label, float(probs[idx]), "Fine-tuned RoBERTa", "Model output is a screening signal, not proof of truth.", (time.perf_counter()-started)*1000)

class SourceIngestionAgent:
    """Ingest article text, text/CSV/PDF files, or public article URLs."""
    def from_url(self, url: str) -> EvidenceChunk:
        parsed = urlparse(url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Enter a valid http(s) article URL.")
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 (compatible; NewsRAG/1.0)"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for node in soup(["script", "style", "nav", "footer", "header", "noscript", "form"]):
            node.decompose()
        article = soup.find("article") or soup.find("main") or soup.body or soup
        text = " ".join(article.stripped_strings)
        if len(text) < 80:
            raise ValueError("Could not extract enough article text from this URL. Paste the article text or upload a file instead.")
        title = soup.title.get_text(" ", strip=True) if soup.title else parsed.netloc
        return EvidenceChunk(f"{title} — {url}", text, 0)

    def from_text(self, text: str, source: str = "Pasted article") -> List[EvidenceChunk]:
        text = re.sub(r"\s+", " ", (text or "")).strip()
        if len(text) < 30:
            raise ValueError("Provide at least 30 characters of article/source text.")
        words = text.split()
        chunks, current, count = [], [], 0
        # Roughly 180-word chunks with 30-word overlap.
        while count < len(words):
            end = min(count + 180, len(words))
            chunk = " ".join(words[count:end])
            chunks.append(EvidenceChunk(source, chunk, len(chunks)))
            if end == len(words): break
            count = max(end - 30, count + 1)
        return chunks

    def from_file(self, filename: str, data: bytes) -> List[EvidenceChunk]:
        low = filename.lower()
        if low.endswith(".txt") or low.endswith(".md"):
            text = data.decode("utf-8", errors="replace")
            return self.from_text(text, filename)
        if low.endswith(".pdf"):
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return self.from_text(text, filename)
        if low.endswith(".csv"):
            import io
            df = pd.read_csv(io.BytesIO(data))
            text_cols = [c for c in df.columns if any(k in str(c).lower() for k in ["text", "article", "content", "news", "title", "description"])]
            if not text_cols:
                text_cols = [c for c in df.columns if df[c].dtype == object]
            if not text_cols: raise ValueError("No text-like columns found in CSV.")
            chunks = []
            for i, row in df.iterrows():
                row_text = " | ".join(f"{c}: {str(row[c])}" for c in text_cols if pd.notna(row[c]))
                if len(row_text.strip()) >= 30:
                    chunks.extend(self.from_text(row_text, f"{filename} row {i+1}"))
            return chunks
        raise ValueError("Supported file types: TXT, MD, PDF, CSV.")

class RetrievalAgent:
    """Local TF-IDF RAG retriever. It retrieves only from sources added by the user."""
    def __init__(self):
        self.chunks: List[EvidenceChunk] = []
        self.vectorizer = None
        self.matrix = None

    def add_chunks(self, chunks: List[EvidenceChunk]):
        self.chunks.extend(chunks)
        self._rebuild()

    def clear(self):
        self.chunks = []; self.vectorizer = None; self.matrix = None

    def _rebuild(self):
        texts = [c.text for c in self.chunks if c.text.strip()]
        if not texts:
            self.vectorizer = None; self.matrix = None; return
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=50000, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, question: str, k: int = 4) -> List[Dict[str, Any]]:
        if not self.chunks or self.vectorizer is None:
            return []
        qv = self.vectorizer.transform([question])
        scores = cosine_similarity(qv, self.matrix).ravel()
        indices = scores.argsort()[::-1][:max(1,k)]
        results = []
        for i in indices:
            if scores[i] <= 0: continue
            c = self.chunks[int(i)]
            results.append({"source": c.source, "text": c.text, "score": float(scores[i]), "chunk_id": c.chunk_id})
        return results

class ExplanationAgent:
    """Optional Ollama explanation, strictly grounded in retrieved evidence."""
    def __init__(self, model: str = "llama3.2"):
        self.model = model

    def answer(self, question: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not evidence:
            return {"answer": "I don't have relevant source material in the current knowledge base. Add an article, URL, PDF, TXT, or CSV first.", "mode": "No evidence available", "sources": []}
        context = "\n\n".join(f"[Source {i+1}: {e['source']} | relevance {e['score']:.3f}]\n{e['text']}" for i,e in enumerate(evidence))
        prompt = f"""You are a careful news research assistant. Answer the user's question using ONLY the evidence below. Distinguish claims in the article from verified facts. If evidence is insufficient, say so. Do not invent dates, events, quotes, or sources. Include source markers such as [Source 1] in the answer. Explain uncertainty and do not call a story true or false based solely on writing style.\n\nEVIDENCE:\n{context}\n\nQUESTION: {question}\n\nAnswer in concise, clear language."""
        try:
            import ollama
            response = ollama.chat(model=self.model, messages=[{"role":"user", "content":prompt}], options={"temperature":0.1})
            answer = response["message"]["content"].strip()
            return {"answer": answer, "mode": f"Ollama ({self.model}) + retrieved evidence", "sources": list(dict.fromkeys(e["source"] for e in evidence))}
        except Exception as exc:
            # Evidence extract fallback remains useful if Ollama isn't installed/running.
            snippets = "\n\n".join(f"**{e['source']}** (relevance {e['score']:.2f})\n{e['text'][:900]}" for e in evidence)
            return {"answer": "Ollama is unavailable, so no generated answer was produced. Here are the most relevant retrieved passages instead. Start Ollama and pull the configured model for a synthesized answer.\n\n" + snippets, "mode": f"Retrieval-only fallback ({type(exc).__name__})", "sources": list(dict.fromkeys(e["source"] for e in evidence))}

class NewsAgentOrchestrator:
    def __init__(self, model_dir: str = "models/roberta-fake-news"):
        self.classifier = NewsClassifierAgent(model_dir)
        self.ingestion = SourceIngestionAgent()
        self.retriever = RetrievalAgent()
        self.explainer = ExplanationAgent(os.getenv("OLLAMA_MODEL", "llama3.2"))

    def answer_question(self, question: str, top_k: int = 4):
        evidence = self.retriever.retrieve(question, top_k)
        return self.explainer.answer(question, evidence), evidence
