"""Amendment-aware hybrid retrieval.

  VectorIndex  - ChromaDB persistence + dense (cosine) search with metadata filters.
  Retriever    - 1) select documents applicable to the requested assessment period (document level)
                 2) dense search + TF-IDF lexical search over the allowed chunks, fused with RRF
                 3) relevance floor -> explicit "insufficient evidence" instead of weak context
                 4) version flags (mixed law cut-offs, historical material) and numeric conflict report
Legal text is full of exact tokens ("section 17(5)", "180 days", "Rule 36(4)") that dense MiniLM
embeddings blur; the lexical channel recovers them. RRF needs no score calibration between channels.
"""
from __future__ import annotations

import hashlib
import re
from functools import cached_property
from pathlib import Path

import numpy as np

from .config import load_yaml
from .schemas import Chunk, GenerationRequest, RetrievalResult, RetrievedChunk, SourceDocument

RRF_K = 60
NUM_RE = re.compile(r"(?:₹|rs\.?\s?)?\d[\d,]*(?:\.\d+)?\s?(?:%|per\s?cent|days?|months?|years?|lakhs?|crores?)?", re.I)


# --------------------------------------------------------------------------- embeddings
class OnnxEmbedder:
    """all-MiniLM-L6-v2 via ChromaDB's bundled ONNX runtime (no torch). 384-d, truncates at 256 word-pieces."""

    name = "onnx-all-MiniLM-L6-v2"

    @cached_property
    def _ef(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        return DefaultEmbeddingFunction()

    def __call__(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.asarray(self._ef(texts), dtype=np.float32))


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder. For tests/offline only - not semantically meaningful."""

    name = "hash-256"

    def __call__(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1.0
        return _normalise(out)


def _normalise(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(n == 0, 1, n)


def make_embedder(name: str = "onnx"):
    return HashEmbedder() if name == "hash" else OnnxEmbedder()


def embed_text(c: Chunk) -> str:
    # contextual header: the chunk's chapter/section travel with it into the embedding
    return f"{c.chapter} | {c.section}\n{c.text}"


# --------------------------------------------------------------------------- vector store
class VectorIndex:
    def __init__(self, path: Path, embedder, collection: str = "examforge"):
        import chromadb

        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.embedder = embedder
        self.col = self.client.get_or_create_collection(
            f"{collection}_{embedder.name}".replace(".", "_")[:60], metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], batch: int = 64) -> None:
        for i in range(0, len(chunks), batch):
            part = chunks[i : i + batch]
            self.col.upsert(
                ids=[c.chunk_id for c in part],
                documents=[c.text for c in part],
                embeddings=self.embedder([embed_text(c) for c in part]).tolist(),
                metadatas=[c.model_dump(exclude={"chunk_id", "text"}) for c in part],
            )

    def delete_document(self, document_id: str) -> None:
        self.col.delete(where={"document_id": document_id})

    def count(self) -> int:
        return self.col.count()

    def chunks(self, document_ids: list[str] | None = None) -> list[Chunk]:
        where = {"document_id": {"$in": document_ids}} if document_ids else None
        if document_ids == []:
            return []
        res = self.col.get(where=where, include=["documents", "metadatas"])
        return [Chunk(chunk_id=i, text=d, **m) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]

    def dense_search(self, query: str, document_ids: list[str], k: int, where_extra: dict | None = None) -> list[tuple[Chunk, float]]:
        if not document_ids:
            return []
        conds = [{"document_id": {"$in": document_ids}}] + ([where_extra] if where_extra else [])
        where = conds[0] if len(conds) == 1 else {"$and": conds}
        n = min(k, self.col.count())
        if n == 0:
            return []
        res = self.col.query(
            query_embeddings=self.embedder([query]).tolist(), n_results=n, where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            (Chunk(chunk_id=i, text=d, **m), 1.0 - dist)
            for i, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def embeddings(self, ids: list[str]) -> dict[str, np.ndarray]:
        res = self.col.get(ids=ids, include=["embeddings"])
        return {i: np.asarray(e) for i, e in zip(res["ids"], res["embeddings"])}


# --------------------------------------------------------------------------- source selection (amendment-aware)
def select_documents(docs: list[SourceDocument], req: GenerationRequest) -> tuple[list[str], list[str], list[str]]:
    """Return (allowed_doc_ids, historical_doc_ids, warnings). Selection is by stated applicability,
    never by recency: a newer document is not assumed to supersede an older one."""
    usable = [
        d for d in docs
        if d.subject == req.subject and (d.extraction_status or "").startswith("ok")
        and (not req.document_types or d.document_type in req.document_types)
    ]
    warnings: list[str] = []
    if req.attempt is None:
        periods = sorted({a for d in usable for a in d.attempts()})
        if len(periods) > 1:
            warnings.append(f"No assessment period selected: sources from {periods} may be mixed. Verify applicability.")
        return [d.document_id for d in usable], [], warnings
    current = [d.document_id for d in usable if req.attempt in d.attempts()]
    historical = [d.document_id for d in usable if req.attempt not in d.attempts()] if req.allow_historical else []
    if historical:
        warnings.append(
            f"Historical material intentionally included ({len(historical)} document(s) not tagged for {req.attempt}); "
            "provisions may have changed - human verification required."
        )
    return current + historical, historical, warnings


# --------------------------------------------------------------------------- retriever
class Retriever:
    def __init__(self, index: VectorIndex, docs: list[SourceDocument], top_k: int = 6, min_relevance: float = 0.25):
        self.index, self.docs, self.top_k, self.min_relevance = index, docs, top_k, min_relevance
        self._lex = None

    def _lexical(self):
        if self._lex is None:
            from sklearn.feature_extraction.text import TfidfVectorizer

            chunks = self.index.chunks()
            # token pattern keeps section refs like 17(5) and 16(4) as tokens
            vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b\w[\w()]*\b", min_df=1)
            mat = vec.fit_transform([embed_text(c) for c in chunks]) if chunks else None
            self._lex = (chunks, vec, mat)
        return self._lex

    def build_query(self, req: GenerationRequest) -> str:
        topics = load_yaml("subjects.yaml").get("subjects", {}).get(req.subject, {}).get("chapters", {})
        kw = (topics.get(req.chapter, {}).get("topics", {}) or {}).get(req.topic, []) or []
        mode = " amendment inserted substituted with effect from Finance Act" if req.mode == "amendment" else ""
        return f"{req.topic}. {req.chapter}. {' '.join(kw)}{mode}".strip()

    def retrieve(self, req: GenerationRequest, query: str | None = None, method: str = "hybrid", k: int | None = None) -> RetrievalResult:
        k = k or self.top_k
        query = query or self.build_query(req)
        allowed, historical, warnings = select_documents(self.docs, req)
        # chapter filter: chapter-specific docs must match; multi-chapter papers (RTP/MTP, chapter None) stay eligible
        by_id = {d.document_id: d for d in self.docs}
        allowed = [i for i in allowed if not by_id[i].chapter or by_id[i].chapter == req.chapter]
        filters = {"subject": req.subject, "chapter": req.chapter, "attempt": req.attempt,
                   "allow_historical": req.allow_historical, "documents": allowed, "mode": req.mode}
        if not allowed:
            avail = sorted({a for d in self.docs if d.subject == req.subject for a in d.attempts()})
            return RetrievalResult(query=query, filters=filters, chunks=[], warnings=warnings, insufficient=True,
                                   reason=f"No indexed source documents for chapter '{req.chapter}' applicable to "
                                          f"{req.attempt or 'any period'}. Available periods: {avail or 'none'}.")
        where_extra = {"amendment_marker": True} if req.mode == "amendment" else None
        pool = max(25, 4 * k)

        dense = self.index.dense_search(query, allowed, pool, where_extra) if method in ("hybrid", "dense") else []
        lexical: list[tuple[Chunk, float]] = []
        if method in ("hybrid", "lexical"):
            chunks, vec, mat = self._lexical()
            if mat is not None:
                allowed_set = set(allowed)
                scores = (mat @ vec.transform([query]).T).toarray().ravel()
                order = [i for i in np.argsort(-scores) if chunks[i].document_id in allowed_set
                         and scores[i] > 0 and (not where_extra or chunks[i].amendment_marker)]
                lexical = [(chunks[i], float(scores[i])) for i in order[:pool]]

        fused: dict[str, RetrievedChunk] = {}
        for rank, (c, s) in enumerate(dense):
            fused[c.chunk_id] = RetrievedChunk(chunk=c, score=1 / (RRF_K + rank + 1), dense_similarity=round(s, 4))
        for rank, (c, s) in enumerate(lexical):
            rc = fused.setdefault(c.chunk_id, RetrievedChunk(chunk=c, score=0.0))
            rc.score += 1 / (RRF_K + rank + 1)
            rc.lexical_score = round(s, 4)
        ranked = sorted(fused.values(), key=lambda r: -r.score)[:k]

        result = RetrievalResult(query=query, filters=filters, chunks=ranked, warnings=warnings)
        best_dense = max((s for _, s in dense), default=None)
        if not ranked:
            result.insufficient, result.reason = True, (
                "No documented amendments found in the selected sources for this chapter."
                if req.mode == "amendment" else "Retrieval returned no chunks.")
        elif method != "lexical" and (best_dense is None or best_dense < self.min_relevance):
            result.insufficient = True
            result.reason = f"Best semantic match {best_dense if best_dense is not None else 'n/a'} is below the relevance floor {self.min_relevance}."
        hist = set(historical)
        if any(r.chunk.document_id in hist for r in ranked):
            result.warnings.append("Context includes chunks from material not tagged for the selected period.")
        cutoffs = {r.chunk.law_cutoff for r in ranked if r.chunk.law_cutoff}
        if len(cutoffs) > 1:
            result.warnings.append(f"Context mixes documents with different law cut-offs: {sorted(cutoffs)}.")
        result.conflicts = detect_conflicts(ranked, self.index)
        return result


def _numbers(text: str) -> set[str]:
    return {re.sub(r"\s+", "", m.group(0).lower()) for m in NUM_RE.finditer(text) if any(ch.isdigit() for ch in m.group(0))}


def detect_conflicts(ranked: list[RetrievedChunk], index: VectorIndex | None, sim_threshold: float = 0.8) -> list[dict]:
    """Heuristic source-conflict report: two chunks from *different documents* that are about the same
    thing (high embedding similarity) but quote different numeric facts (rates, limits, periods).
    A flag means "a human must check which version applies", not "the newer one wins"."""
    if index is None or len(ranked) < 2:
        return []
    embs = index.embeddings([r.chunk.chunk_id for r in ranked])
    out = []
    for i, a in enumerate(ranked):
        for b in ranked[i + 1 :]:
            ca, cb = a.chunk, b.chunk
            if ca.document_id == cb.document_id or ca.chunk_id not in embs or cb.chunk_id not in embs:
                continue
            sim = float(np.dot(embs[ca.chunk_id], embs[cb.chunk_id]))
            if sim < sim_threshold:
                continue
            na, nb = _numbers(ca.text), _numbers(cb.text)
            only_a, only_b = na - nb, nb - na
            if only_a and only_b:
                out.append({
                    "chunk_a": ca.chunk_id, "chunk_b": cb.chunk_id, "similarity": round(sim, 3),
                    "doc_a": f"{ca.title} ({ca.attempt}, cut-off {ca.law_cutoff or 'n/a'})",
                    "doc_b": f"{cb.title} ({cb.attempt}, cut-off {cb.law_cutoff or 'n/a'})",
                    "numbers_only_in_a": sorted(only_a)[:8], "numbers_only_in_b": sorted(only_b)[:8],
                })
    return out
