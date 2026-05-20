from typing import List, Dict, Tuple, Optional, Any
from sentence_transformers import CrossEncoder
from components.pre_retrieval import Chunk

class Reranker:
    MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

    def __init__(self, top_k: int=3, score_threshold: float=-5.0):
        print(f"[Reranker] Loading '{self.MODEL_NAME}' …")
        self.model = CrossEncoder(self.MODEL_NAME)
        self.top_k = top_k
        self.threshold = score_threshold
        print('[Reranker] Ready.')

    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        if not chunks:
            return []
        pairs = [(query, c.content) for c in chunks]
        scores = self.model.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk.metadata['rerank_score'] = float(score)
        ranked = sorted(chunks, key=lambda c: c.metadata['rerank_score'], reverse=True)
        ranked = [c for c in ranked if c.metadata['rerank_score'] >= self.threshold]
        ranked = ranked[:self.top_k]
        print(f'[Reranker] {len(chunks)} → {len(ranked)} chunks (threshold={self.threshold}, top_k={self.top_k})')
        return ranked

class ContextualCompressor:
    _COMPRESS_PROMPT = 'You are a precise text extractor. Given a QUESTION and a DOCUMENT EXCERPT, extract ONLY the sentences from the excerpt that are directly relevant to answering the question.\n\nRules:\n- Return only verbatim sentences from the excerpt. Do not paraphrase.\n- If NO sentence is relevant, output exactly: IRRELEVANT\n- Do not add any explanation, preamble, or punctuation outside the extracted sentences.\n\nQUESTION: {query}\n\nDOCUMENT EXCERPT:\n"""\n{context}\n"""\n\nRelevant sentences:'

    def __init__(self, groq_client, model: str='openai/gpt-oss-120b', max_input_words: int=500):
        self.client = groq_client
        self.model = model
        self.max_input_words = max_input_words

    def _compress_one(self, query: str, chunk: Chunk) -> Optional[Chunk]:
        words = chunk.content.split()
        truncated = ' '.join(words[:self.max_input_words])
        try:
            resp = self.client.chat.completions.create(model=self.model, messages=[{'role': 'user', 'content': self._COMPRESS_PROMPT.format(query=query, context=truncated)}], max_tokens=300, temperature=0.0)
            result = resp.choices[0].message.content.strip()
            if result.upper() == 'IRRELEVANT':
                return None
            chunk.metadata['original_content'] = chunk.content
            chunk.content = result
            return chunk
        except Exception as e:
            print(f'[Compressor] Failed for chunk {chunk.chunk_id}: {e}')
            return chunk

    def compress(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        before = len(chunks)
        compressed = []
        for chunk in chunks:
            result = self._compress_one(query, chunk)
            if result is not None:
                compressed.append(result)
        print(f'[Compressor] {before} → {len(compressed)} chunks after compression ({before - len(compressed)} dropped).')
        return compressed

class PostRetrievalPipeline:

    def __init__(self, groq_client, reranker_top_k: int=3, use_compression: bool=True):
        self.reranker = Reranker(top_k=reranker_top_k)
        self.compressor = ContextualCompressor(groq_client)
        self.use_compression = use_compression

    def run(self, query: str, chunks: List[Chunk]) -> Tuple[List[Chunk], Dict[str, Any]]:
        print('\n' + '═' * 60)
        print('  PHASE 3 — POST-RETRIEVAL')
        print('═' * 60)
        meta: Dict[str, Any] = {'input_chunks': len(chunks)}
        chunks = self.reranker.rerank(query, chunks)
        meta['after_rerank'] = len(chunks)
        if self.use_compression and chunks:
            chunks = self.compressor.compress(query, chunks)
            meta['after_compression'] = len(chunks)
        meta['final_chunks'] = len(chunks)
        print(f'\n[Phase 3 Complete] {len(chunks)} chunks ready for generation.\n')
        return (chunks, meta)
