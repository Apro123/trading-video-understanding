"""
VSS Semantic Video Search Engine.
Implements the Video Search capability of the NVIDIA VSS Blueprint:
https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization

Provides hybrid semantic vector search and exact keyword indexing across
multimodal video chunks (dense VLM captions, visual reasoning, and spoken transcripts).
"""

import os
import re
import math
import json
from typing import List, Dict, Any, Optional, Tuple

from .schemas import (
    VideoChunk,
    VlmCaptionResponse,
    SearchQuery,
    SearchResultItem,
    SearchResponse
)

class VSSSearchIndex:
    """
    In-memory vector and keyword search index for video chunks.
    Compatible with NVIDIA VSS search tools and downstream agent queries.
    """

    def __init__(self, video_id: str = "default_video"):
        self.video_id = video_id
        self.chunks_data: List[Dict[str, Any]] = []
        self._doc_frequencies: Dict[str, int] = {}
        self._total_docs: int = 0
        self._embedding_model = None
        self._has_embeddings = False
        self._init_embedding_model()

    def _init_embedding_model(self):
        """Attempts to load sentence-transformers if available."""
        try:
            from sentence_transformers import SentenceTransformer
            # Lightweight, high-speed 384-dim semantic embedding model
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._has_embeddings = True
        except Exception:
            # Graceful fallback to TF-IDF and keyword matching if sentence-transformers not installed
            self._has_embeddings = False

    def _tokenize(self, text: str) -> List[str]:
        """Simple alphanumeric tokenizer with lowercase normalization."""
        return re.findall(r'[a-z0-9_\.]+', text.lower())

    def index_chunks(
        self,
        chunks: List[VideoChunk],
        captions: List[VlmCaptionResponse]
    ):
        """
        Indexes video chunks with their corresponding VLM dense captions,
        reasoning notes, and spoken subtitles.
        """
        caption_map = {c.chunk_id: c for c in captions}
        self.chunks_data = []
        self._doc_frequencies = {}

        for chunk in chunks:
            cap = caption_map.get(chunk.chunk_id)
            caption_content = cap.content if cap else ""
            reasoning = cap.reasoning_description if cap else ""
            signals = cap.detected_trade_signals if cap else []

            # Fuse multimodal content into a comprehensive searchable document
            searchable_text = f"{chunk.synchronized_transcript} {caption_content} {reasoning} {' '.join(signals)}"

            doc_entry = {
                "chunk_id": chunk.chunk_id,
                "start_s": chunk.start_s,
                "end_s": chunk.end_s,
                "timestamp_range": f"{chunk.start_str} - {chunk.end_str}",
                "transcript": chunk.synchronized_transcript,
                "caption": caption_content,
                "reasoning": reasoning,
                "signals": signals,
                "full_text": searchable_text,
                "tokens": self._tokenize(searchable_text),
                "embedding": None
            }

            # Precompute embedding if model is loaded
            if self._has_embeddings and self._embedding_model:
                try:
                    vec = self._embedding_model.encode(searchable_text, convert_to_numpy=True)
                    doc_entry["embedding"] = vec
                except Exception:
                    pass

            self.chunks_data.append(doc_entry)

        # Build term frequencies for BM25-like scoring
        self._total_docs = len(self.chunks_data)
        for doc in self.chunks_data:
            unique_terms = set(doc["tokens"])
            for t in unique_terms:
                self._doc_frequencies[t] = self._doc_frequencies.get(t, 0) + 1

    def search(self, query: str, top_k: int = 3, min_score: float = 0.0) -> SearchResponse:
        """
        Executes a search query against indexed chunks using hybrid scoring.
        """
        if not self.chunks_data:
            return SearchResponse(query=query, total_chunks=0, results=[])

        q_tokens = self._tokenize(query)
        scored_results: List[Tuple[float, Dict[str, Any]]] = []

        q_embedding = None
        if self._has_embeddings and self._embedding_model:
            try:
                q_embedding = self._embedding_model.encode(query, convert_to_numpy=True)
            except Exception:
                q_embedding = None

        for doc in self.chunks_data:
            # 1. Lexical BM25-style keyword score
            lexical_score = 0.0
            doc_len = len(doc["tokens"]) or 1
            for qt in q_tokens:
                if qt in doc["tokens"]:
                    count = doc["tokens"].count(qt)
                    df = self._doc_frequencies.get(qt, 1)
                    idf = math.log((self._total_docs - df + 0.5) / (df + 0.5) + 1.0)
                    tf = count / float(count + 1.5 * (0.25 + 0.75 * doc_len / 40.0))
                    lexical_score += idf * tf

            # 2. Semantic vector cosine similarity
            vector_score = 0.0
            if q_embedding is not None and doc.get("embedding") is not None:
                try:
                    import numpy as np
                    dot = np.dot(q_embedding, doc["embedding"])
                    norm_q = np.linalg.norm(q_embedding)
                    norm_d = np.linalg.norm(doc["embedding"])
                    if norm_q > 0 and norm_d > 0:
                        vector_score = float(dot / (norm_q * norm_d))
                except Exception:
                    vector_score = 0.0

            # Combined hybrid score (normalized between 0.0 and 1.0+)
            total_score = round(vector_score * 0.7 + min(1.0, lexical_score * 0.3), 4)

            # Boost if exact price number or critical terms match
            for qt in q_tokens:
                if any(c.isdigit() for c in qt) and qt in doc["tokens"]:
                    total_score += 0.25

            if total_score >= min_score:
                scored_results.append((total_score, doc))

        # Sort descending by score
        scored_results.sort(key=lambda x: x[0], reverse=True)
        top_results = scored_results[:top_k]

        items = []
        for score, doc in top_results:
            # Summarize trade context
            signals_desc = ", ".join(doc["signals"]) if doc["signals"] else "Technical Chart Review"
            items.append(SearchResultItem(
                chunk_id=doc["chunk_id"],
                start_s=doc["start_s"],
                end_s=doc["end_s"],
                timestamp_range=doc["timestamp_range"],
                score=round(score, 4),
                matched_caption=doc["caption"],
                matched_transcript=doc["transcript"],
                reasoning=doc["reasoning"][:200] + ("..." if len(doc["reasoning"]) > 200 else ""),
                trade_context=signals_desc
            ))

        return SearchResponse(
            query=query,
            total_chunks=len(self.chunks_data),
            results=items
        )

    def export_index_json(self) -> Dict[str, Any]:
        """Exports index metadata suitable for saving to vss_search_index.json."""
        serialized_chunks = []
        for doc in self.chunks_data:
            serialized_chunks.append({
                "chunk_id": doc["chunk_id"],
                "start_s": doc["start_s"],
                "end_s": doc["end_s"],
                "timestamp_range": doc["timestamp_range"],
                "transcript": doc["transcript"],
                "caption": doc["caption"],
                "reasoning": doc["reasoning"],
                "signals": doc["signals"]
            })
        return {
            "video_id": self.video_id,
            "total_chunks": len(self.chunks_data),
            "chunks": serialized_chunks
        }
