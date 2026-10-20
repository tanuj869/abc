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

        # 3. Ingest Detailed Guidelines, Subsidy Slabs, and Regulatory Knowledge
        guideline_chunks = [
            ("PM_Surya_Ghar_Operational_Guidelines.pdf", 
             "PM Surya Ghar Muft Bijli Yojana Central Financial Assistance (CFA) Subsidy Slabs:\n"
             "• Systems up to 1 kW: ₹30,000 direct capital subsidy credited to beneficiary bank account.\n"
             "• Systems up to 2 kW: ₹60,000 direct capital subsidy.\n"
             "• Systems of 3 kW and higher: ₹78,000 total capital subsidy (capped at ₹78,000 for individual residential households).\n"
             "• Group Housing Societies / Resident Welfare Associations (GHS/RWA): ₹18,000 per kW for common facilities up to 500 kW.\n"
             "• Scheme Outlay: ₹75,021 Crore cabinet approved national expenditure to solarize 1 Crore (10 Million) residential households by 2027."),
             
            ("PM_Surya_Ghar_Operational_Guidelines.pdf",
             "Application and Installation Process on National Portal (pmsuryaghar.gov.in):\n"
             "1. Consumer Registration: Register using electricity consumer account number, mobile number, and DISCOM name.\n"
             "2. Technical Feasibility Approval: DISCOM verifies local distribution transformer capacity within 15 days.\n"
             "3. Installation by Empaneled Vendor: System installed by registered vendor using ALMM-compliant domestic solar cells/modules.\n"
             "4. Net Metering & Inspection: DISCOM inspects installation and installs bi-directional net meter.\n"
             "5. Subsidy Disbursement: CFA subsidy released online directly into the beneficiary's bank account within 30 days of commissioning."),
             
            ("PM_Surya_Ghar_Data_Science_Guideline.pdf", 
             "Core Econometric Problem Statement and Guidelines: Explaining variation in rooftop solar adoption across Indian states and predicting uptake under alternative policy and economic scenarios. Requires building a verified 36-state panel integrating administrative progress, grid tariffs, solar potential, RE capacity, and DISCOM health."),
             
            ("PM_Surya_Ghar_Data_Science_Guideline.pdf", 
             "Scenario Simulation Guidelines: Scenario A (Base Case observed vs fitted), Scenario B (Subsidy shifts +10% to -20%), Scenario C (Sunbelt high-potential targeting for states with >5.2 kWh/m2/day), Scenario D (Electricity tariff shock +10pp to +15pp YoY), and Scenario E (Combined policy intervention)."),
             
            ("PMSG_Research_Report.docx", 
             "Empirical Discovery of the Subsidy Paradox: Cross-sectional econometric modeling proves that baseline electricity tariffs and static technical potential are among the weakest predictors of state rooftop adoption (r < 0.15). In contrast, recent electricity tariff growth (% YoY, r=0.47) and climatological solar irradiance (r=0.56) dominate state-level conversion efficiency."),
             
            ("PMSG_Research_Report.docx", 
             "Diagnostic Root Causes for Adoption Lag in Low-Income States (e.g. Bihar, UP, Odisha):\n"
             "• High DISCOM AT&C losses and weaker operational balance sheets (C/C- ratings) leading to delays in net meter provisioning.\n"
             "• Lower baseline domestic tariffs and extensive state electricity bill cross-subsidies reduce consumer perceived urgency.\n"
             "• High upfront capital expenditure relative to household per-capita income, requiring collateral-free priority lending."),
             
            ("PMSG_Research_Report.docx", 
             "Model Selection and Generalization: A parsimonious 2-variable linear model (tariff growth + irradiance) generalizes better under Leave-One-Out Cross-Validation (MAE 0.135, -20.7% vs naive) than a 6-feature Random Forest (MAE 0.161), demonstrating that complex models overfit on state-level aggregates."),
             
            ("PM_Surya_Ghar_AI_Project_Summary_Report.docx", 
             "Two-Layer Architecture: Divides platform into Layer 1 Core Data Science (data pipelines, feature engineering, forecasting, scenario modeling, interactive dashboard) and Layer 2 AI Intelligence (LangGraph stateful workflow, LangChain tool bindings, Vectorless RAG, Guardrails, and automated evaluation benchmark).")
        ]
        for src, text in guideline_chunks:
            self.corpus.append({
                "doc_name": src,
                "chunk_id": f"{src}#Section",
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
