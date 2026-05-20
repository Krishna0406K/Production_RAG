import json
import numpy as np
from typing import List, Dict, Tuple, Optional
import faiss
from components.pre_retrieval import Chunk, AdvancedEmbedder

class VectorStore:

    def __init__(self, dim: int=384):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Chunk] = []

    def add_chunks(self, chunks: List[Chunk]):
        valid = [c for c in chunks if c.embedding is not None]
        if not valid:
            return
        matrix = np.stack([c.embedding for c in valid]).astype('float32')
        self.index.add(matrix)
        self.chunks.extend(valid)
        print(f'[VectorStore] +{len(valid)} vectors (total={len(self.chunks)})')

    def search(self, query_vec: np.ndarray, top_k: int=5) -> List[Chunk]:
        if not self.chunks:
            return []
        q = query_vec.astype('float32').reshape(1, -1)
        scores, indices = self.index.search(q, min(top_k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                chunk = self.chunks[idx]
                chunk.metadata['retrieval_score'] = float(score)
                results.append(chunk)
        return results

    def get_by_id(self, chunk_id: str) -> Optional[Chunk]:
        for c in self.chunks:
            if c.chunk_id == chunk_id:
                return c
        return None

class QueryOptimizer:
    _REWRITE_PROMPT = 'Rewrite the following search query to be more specific and retrieval-friendly. Return ONLY the rewritten query with no explanation, preamble, or punctuation around it.\n\nOriginal query: {query}'
    _EXPAND_PROMPT = 'Generate {n} distinct alternative phrasings of the following question that would help retrieve relevant documents from a knowledge base. Return ONLY a valid JSON array of strings, no other text.\n\nQuestion: {query}'

    def __init__(self, groq_client, model: str='openai/gpt-oss-120b', n_variants: int=3):
        self.client = groq_client
        self.model = model
        self.n_variants = n_variants

    def _call(self, prompt: str, max_tokens: int=200) -> str:
        resp = self.client.chat.completions.create(model=self.model, messages=[{'role': 'user', 'content': prompt}], max_tokens=max_tokens, temperature=0.3)
        return resp.choices[0].message.content.strip()

    def rewrite(self, query: str) -> str:
        try:
            rewritten = self._call(self._REWRITE_PROMPT.format(query=query))
            print(f"[QueryOptimizer] Rewritten: '{query[:50]}' → '{rewritten[:60]}'")
            return rewritten
        except Exception as e:
            print(f'[QueryOptimizer] Rewrite failed ({e}), using original.')
            return query

    def expand(self, query: str) -> List[str]:
        try:
            raw = self._call(self._EXPAND_PROMPT.format(n=self.n_variants, query=query), max_tokens=350)
            raw = raw.replace('```json', '').replace('```', '').strip()
            variants = json.loads(raw)
            if isinstance(variants, list) and variants:
                print(f'[QueryOptimizer] Expanded to {len(variants)} variants.')
                return [str(v) for v in variants]
        except Exception as e:
            print(f'[QueryOptimizer] Expansion failed ({e}), using original.')
        return [query]

class FusionRetriever:

    def __init__(self, vector_store: VectorStore, query_optimizer: QueryOptimizer, embedder: AdvancedEmbedder, top_k: int=5, rrf_k: int=60):
        self.store = vector_store
        self.optimizer = query_optimizer
        self.embedder = embedder
        self.top_k = top_k
        self.rrf_k = rrf_k

    def _rrf_merge(self, result_lists: List[List[Chunk]]) -> List[Chunk]:
        rrf_scores: Dict[str, float] = {}
        chunk_index: Dict[str, Chunk] = {}
        for result_list in result_lists:
            for rank, chunk in enumerate(result_list, start=1):
                cid = chunk.chunk_id
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rank + self.rrf_k)
                chunk_index[cid] = chunk
        top_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
        merged = []
        for cid in top_ids[:self.top_k]:
            chunk = chunk_index[cid]
            chunk.metadata['rrf_score'] = round(rrf_scores[cid], 6)
            merged.append(chunk)
        return merged

    def retrieve(self, query: str) -> List[Chunk]:
        variants = [query] + self.optimizer.expand(query)
        print(f'[FusionRetriever] Searching {len(variants)} query variants …')
        per_variant_results: List[List[Chunk]] = []
        for variant in variants:
            q_vec = self.embedder.embed_query(variant)
            results = self.store.search(q_vec, top_k=self.top_k * 2)
            per_variant_results.append(results)
        merged = self._rrf_merge(per_variant_results)
        print(f'[FusionRetriever] Merged → {len(merged)} unique chunks.')
        return merged

class RetrievalPipeline:

    def __init__(self, flat_store: VectorStore, fine_store: VectorStore, coarse_store: VectorStore, embedder: AdvancedEmbedder, groq_client, top_k: int=5):
        self.flat_store = flat_store
        self.fine_store = fine_store
        self.coarse_store = coarse_store
        self.embedder = embedder
        self.top_k = top_k
        self.optimizer = QueryOptimizer(groq_client, n_variants=3)
        self.fusion = FusionRetriever(flat_store, self.optimizer, embedder, top_k=top_k)

    def retrieve(self, query: str) -> Tuple[List[Chunk], Dict]:
        print('\n' + '═' * 60)
        print('  PHASE 2 — RETRIEVAL')
        print('═' * 60)
        refined_query = self.optimizer.rewrite(query)
        chunks = self.fusion.retrieve(refined_query)
        print(f'\n[Phase 2 Complete] Retrieved {len(chunks)} chunks.\n')
        return (chunks, {'strategy': 'fusion_rrf', 'chunks_retrieved': len(chunks)})
