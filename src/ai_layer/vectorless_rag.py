"""
Vectorless RAG Engine:
Implements classical BM25 ranking, keyword search, and metadata filtering over
project policy reports, dataset dictionaries, and research documentation.
No external vector database required (zero-cost, lightweight, reproducible).
"""

import os
import sys
import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

class VectorlessRAG:
    def __init__(self, workspace_dir: str = _ROOT):
        self.workspace_dir = workspace_dir
        self.corpus: List[Dict[str, Any]] = []
        self.tokenized_corpus = []
        self.bm25 = None
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        # Simple whitespace & alphanumeric tokenization with lowercasing
        tokens = re.findall(r'\b[a-zA-Z0-9_\u20b9\-]+\b', text.lower())
        return tokens

    def _add_document_chunks(self, doc_name: str, text: str, chunk_size: int = 150):
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for idx, p in enumerate(paragraphs):
            words = p.split()
            if len(words) < 8:
                continue
            self.corpus.append({
                "doc_name": doc_name,
                "chunk_id": f"{doc_name}#P{idx+1}",
                "text": p
            })

    def _build_index(self):
        # 1. Ingest DATA_DICTIONARY.md
        dd_path = os.path.join(self.workspace_dir, "DATA_DICTIONARY.md")
        if os.path.exists(dd_path):
            with open(dd_path, "r", encoding="utf-8") as f:
                self._add_document_chunks("DATA_DICTIONARY.md", f.read())

        # 2. Ingest README.md
        readme_path = os.path.join(self.workspace_dir, "README.md")
        if os.path.exists(readme_path):
            with open(readme_path, "r", encoding="utf-8") as f:
                self._add_document_chunks("README.md", f.read())

        # 3. Ingest Guidelines (from extracted text)
        guideline_chunks = [
            ("pm_surya_ghar_data_science_guideline.pdf", "PM Surya Ghar Data Science Project Guidelines: Core problem statement asks what explains variation in adoption across Indian states, and how uptake can be forecast under alternative policy scenarios. Requires building a clean state-level panel, testing external covariates (tariffs, solar potential, RE capacity, demand), and executing scenario simulations."),
            ("pm_surya_ghar_data_science_guideline.pdf", "Guideline Section 5 & 6: Minimum forecasting benchmarks must include Naive, Holt-Winters exponential smoothing, ARIMA/SARIMAX, and Log-linear trend models. At least five representative state case studies must be evaluated with rolling-origin evaluation."),
            ("pm_surya_ghar_data_science_guideline.pdf", "Guideline Section 7: Scenarios include Scenario A (Base Case), Scenario B (Subsidy +10%), Scenario C (High solar potential targeting), Scenario D (Tariff shock +10pp to +15pp), and Scenario E (Combined intervention). Clearly separate predictive scenarios from causal claims."),
            ("PMSG_Research_Report.docx", "Empirical Findings on the Subsidy Paradox: Cross-sectional econometric modeling proves that baseline electricity tariff levels and static potential are weak predictors of rooftop solar adoption. In contrast, recent electricity tariff growth (% YoY, r=0.47) and climatological solar irradiance (r=0.56) dominate state-level conversion efficiency."),
            ("PMSG_Research_Report.docx", "Model Selection Results: A parsimonious 2-variable linear model (tariff growth + irradiance) generalizes better under Leave-One-Out Cross-Validation (MAE 0.135, -20.7% vs naive) than a 6-feature Random Forest (MAE 0.161), demonstrating that complex models overfit on state-level aggregates."),
            ("PM_Surya_Ghar_AI_Project_Summary_Report.docx", "Two-Layer System Architecture: Recommended approach divides the platform into Core Data Science Layer (data engineering, feature engineering, forecasting, scenario modeling, dashboard) and AI Intelligence Layer (LangGraph orchestration, LangChain tooling, Vectorless RAG, Guardrails, and Evals).")
        ]
        for src, text in guideline_chunks:
            self.corpus.append({
                "doc_name": src,
                "chunk_id": f"{src}#Summary",
                "text": text
            })

        # Tokenize and build BM25
        self.tokenized_corpus = [self._tokenize(doc["text"]) for doc in self.corpus]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k most relevant evidence passages using BM25 ranking."""
        if not self.bm25 or not self.corpus:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "doc_name": self.corpus[idx]["doc_name"],
                    "chunk_id": self.corpus[idx]["chunk_id"],
                    "score": round(float(scores[idx]), 3),
                    "text": self.corpus[idx]["text"]
                })
        return results

# Singleton instance
rag_engine = VectorlessRAG()

def get_rag_engine() -> VectorlessRAG:
    """Return singleton VectorlessRAG instance."""
    return rag_engine
