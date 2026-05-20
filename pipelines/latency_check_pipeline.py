import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
import os
import time
import logging
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from groq import Groq
from components.pre_retrieval import PreRetrievalPipeline, Chunk, AdvancedEmbedder
from components.retrieval import RetrievalPipeline, VectorStore
from components.post_retrieval import PostRetrievalPipeline
from components.generation import GenerationPipeline
from components.document_loader import load_document, load_documents
from components.rate_limiter import GroqRateLimiter
from components.latency_logger import LatencyLogger
load_dotenv()
groq_model_api_key = os.getenv('GROQ_API_KEY')
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)-20s  %(levelname)s  %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('rag.pipeline')

def _direct_llm(groq_client, query: str, limiter: GroqRateLimiter) -> str:
    limiter.wait('direct_llm')
    resp = groq_client.chat.completions.create(model='openai/gpt-oss-120b', messages=[{'role': 'user', 'content': query}], max_tokens=300, temperature=0.4)
    return resp.choices[0].message.content.strip()
from components.retrieval import QueryOptimizer

class RateLimitedQueryOptimizer(QueryOptimizer):

    def __init__(self, groq_client, limiter: GroqRateLimiter, **kwargs):
        super().__init__(groq_client, **kwargs)
        self.limiter = limiter

    def _call(self, prompt: str, max_tokens: int=200) -> str:
        if 'alternative phrasings' in prompt:
            self.limiter.wait('query_expand')
        else:
            self.limiter.wait('query_rewrite')
        return super()._call(prompt, max_tokens)
from components.post_retrieval import ContextualCompressor

class RateLimitedCompressor(ContextualCompressor):

    def __init__(self, groq_client, limiter: GroqRateLimiter, **kwargs):
        super().__init__(groq_client, **kwargs)
        self.limiter = limiter

    def _compress_one(self, query: str, chunk: Chunk):
        self.limiter.wait('compress_chunk')
        return super()._compress_one(query, chunk)
from components.generation import ChainOfThoughtGenerator, GenerationPipeline

class RateLimitedCoTGenerator(ChainOfThoughtGenerator):

    def __init__(self, groq_client, limiter: GroqRateLimiter, **kwargs):
        super().__init__(groq_client, **kwargs)
        self.limiter = limiter

    def generate(self, query: str, chunks: List[Chunk]) -> str:
        self.limiter.wait('cot_generation')
        return super().generate(query, chunks)

class RateLimitedGenerationPipeline(GenerationPipeline):

    def __init__(self, groq_client, limiter: GroqRateLimiter):
        self.cot = RateLimitedCoTGenerator(groq_client, limiter)

class AdvancedRAGPipeline:

    def __init__(self, groq_api_key: Optional[str]=None, use_compression: bool=True, use_hierarchical: bool=True, reranker_top_k: int=3, retrieval_top_k: int=8, groq_tpm_limit: int=6000, log_dir: str='logs', verbose_logs: bool=True):
        api_key = groq_model_api_key
        if not api_key:
            raise ValueError("Groq API key is required.\nEither pass groq_api_key='gsk_...' or set GROQ_API_KEY in .env")
        self.groq = Groq(api_key=api_key)
        self.use_compression = use_compression
        self.use_hierarchical = use_hierarchical
        self.retrieval_top_k = retrieval_top_k
        self._reranker_top_k = reranker_top_k
        self.limiter = GroqRateLimiter(tokens_per_minute=groq_tpm_limit)
        self.query_logger = LatencyLogger(log_dir=log_dir, verbose=verbose_logs)
        print('\n▶ Initialising Advanced RAG Pipeline …')
        self.pre_retrieval = PreRetrievalPipeline()
        self.embedder = self.pre_retrieval.embedder
        dim = self.embedder.dim
        self.flat_store = VectorStore(dim=dim)
        self.fine_store = VectorStore(dim=dim)
        self.coarse_store = VectorStore(dim=dim)
        self._retrieval: Optional[RetrievalPipeline] = None
        self._post: Optional[PostRetrievalPipeline] = None
        self._generation: Optional[RateLimitedGenerationPipeline] = None
        print('▶ Pipeline ready. Call .ingest() to add documents.\n')

    def ingest(self, source: str) -> Dict[str, int]:
        doc = load_document(source)
        text = doc['text']
        source_name = doc['source']
        print(f"\n{'═' * 60}\n  INGESTING: '{source_name}'\n{'═' * 60}")
        index_data = self.pre_retrieval.run(text, source=source_name, use_hierarchical=self.use_hierarchical)
        self.flat_store.add_chunks(index_data['flat'])
        if self.use_hierarchical:
            self.fine_store.add_chunks(index_data['fine'])
            self.coarse_store.add_chunks(index_data['coarse'])
        self._build()
        summary = {'flat': len(index_data['flat']), 'fine': len(index_data['fine']), 'coarse': len(index_data['coarse'])}
        print(f"\n✅ Ingested '{source_name}': {summary}")
        return summary

    def ingest_many(self, sources: List[str]) -> List[Dict[str, int]]:
        return [self.ingest(s) for s in sources]

    def _build(self):
        rl_optimizer = RateLimitedQueryOptimizer(self.groq, self.limiter, n_variants=3)
        self._retrieval = RetrievalPipeline(flat_store=self.flat_store, fine_store=self.fine_store, coarse_store=self.coarse_store, embedder=self.embedder, groq_client=self.groq, top_k=self.retrieval_top_k)
        self._retrieval.optimizer = rl_optimizer
        self._retrieval.fusion.optimizer = rl_optimizer
        rl_compressor = RateLimitedCompressor(self.groq, self.limiter)
        self._post = PostRetrievalPipeline(groq_client=self.groq, reranker_top_k=self._reranker_top_k, use_compression=self.use_compression)
        self._post.compressor = rl_compressor
        self._generation = RateLimitedGenerationPipeline(self.groq, self.limiter)

    def query(self, query: str) -> Dict[str, Any]:
        if not self._retrieval:
            raise RuntimeError('No documents ingested. Call .ingest(source) first.')
        print(f'\n{'╔' + '═' * 58 + '╗'}')
        print(f'║  QUERY: {query[:50]:<50}║')
        print(f'{'╚' + '═' * 58 + '╝'}')
        log = self.query_logger.start_query(query)
        self.limiter.reset_stats()
        try:
            log.t_phase2_start = time.perf_counter()
            retrieved_chunks, retrieval_meta = self._retrieval.retrieve(query)
            log.t_phase2_end = time.perf_counter()
            log.chunks_retrieved = len(retrieved_chunks)
            if hasattr(self._retrieval.optimizer, '_last_tokens'):
                log.tokens_query_rewrite = self._retrieval.optimizer._last_tokens
            if not retrieved_chunks:
                print('[Pipeline] No chunks retrieved. Falling back to direct LLM.')
                answer = _direct_llm(self.groq, query, self.limiter)
                log.strategy = 'direct_llm_fallback'
                log.fallback_used = True
                log.rate_limit_wait_secs = self.limiter.status()['total_wait_secs']
                self.query_logger.finish(log, answer=answer)
                return {'answer': answer, 'strategy': 'direct_llm_fallback', 'chunks_used': 0, 'retrieval_meta': retrieval_meta, 'post_meta': {}, 'latency': {'total': log.total_latency()}, 'tokens': {}, 'rate_limit_wait': log.rate_limit_wait_secs}
            log.t_phase3_start = time.perf_counter()
            final_chunks, post_meta = self._post.run(query, retrieved_chunks)
            log.t_phase3_end = time.perf_counter()
            log.chunks_after_rerank = post_meta.get('after_rerank', 0)
            log.chunks_after_compress = post_meta.get('after_compression', post_meta.get('after_rerank', 0))
            if not final_chunks:
                print('[Pipeline] All chunks filtered. Using pre-compression chunks.')
                final_chunks = retrieved_chunks[:2]
                log.fallback_used = True
            log.chunks_used_in_llm = len(final_chunks)
            if hasattr(self._post.compressor, '_last_total_tokens'):
                log.tokens_compress_total = self._post.compressor._last_total_tokens
            log.t_phase4_start = time.perf_counter()
            gen_result = self._generation.generate(query, final_chunks)
            log.t_phase4_end = time.perf_counter()
            log.strategy = gen_result['strategy']
        except Exception as e:
            log.error = str(e)
            log.t_total_end = time.perf_counter()
            self.query_logger.finish(log)
            raise
        rl_status = self.limiter.status()
        log.rate_limit_wait_secs = rl_status['total_wait_secs']
        self.query_logger.finish(log, answer=gen_result['answer'])
        return {'answer': gen_result['answer'], 'strategy': gen_result['strategy'], 'chunks_used': len(final_chunks), 'retrieval_meta': retrieval_meta, 'post_meta': post_meta, 'latency': {'phase2_secs': log.phase2_latency(), 'phase3_secs': log.phase3_latency(), 'phase4_secs': log.phase4_latency(), 'total_secs': log.total_latency()}, 'tokens': {'query_rewrite': log.tokens_query_rewrite, 'compression': log.tokens_compress_total, 'generation': log.tokens_generation, 'total': log.tokens_total}, 'rate_limit_wait': log.rate_limit_wait_secs}

    def ask(self, question: str) -> str:
        return self.query(question)['answer']

    def show_logs(self):
        from components.latency_logger import load_logs
        return load_logs(str(self.query_logger.log_path))
_DEMO_TEXT = '\nRetrieval-Augmented Generation (RAG) is a framework that enhances large language\nmodel responses by grounding them in external knowledge retrieved at inference time.\nInstead of relying solely on knowledge baked into model weights during training,\nRAG retrieves relevant documents from a corpus and feeds them as context to the LLM.\n\nLarge language models can hallucinate facts not in their training data and cannot\naccess information beyond their training cutoff. RAG addresses both: retrieved\ndocuments provide fresh, factual grounding while the LLM handles synthesis and\nnatural language generation.\n\nThe simplest RAG pipeline has three stages: indexing (chunking + embedding),\nretrieval (nearest-neighbour search), and generation (LLM with context).\n\nAdvanced RAG extends the pipeline with hierarchical indexes, query expansion,\nfusion retrieval with Reciprocal Rank Fusion, cross-encoder reranking,\ncontextual compression, and chain-of-thought generation.\n'
if __name__ == '__main__':
    print('=' * 60)
    print('  ADVANCED RAG PIPELINE — DEMO')
    print('=' * 60)
    rag = AdvancedRAGPipeline(use_compression=True, use_hierarchical=True, reranker_top_k=3, retrieval_top_k=8, groq_tpm_limit=6000, log_dir='logs', verbose_logs=True)
    rag.ingest(_DEMO_TEXT)
    questions = ['What is RAG and why is it useful?', 'What are the three stages of the core RAG pipeline?']
    for i, q in enumerate(questions, 1):
        print(f'\n{'─' * 60}\nQ{i}: {q}\n{'─' * 60}')
        result = rag.query(q)
        print(f'\nANSWER:\n{result['answer']}')
        print(f'\nLatency  : {result['latency']}')
        print(f'Tokens   : {result['tokens']}')
        print(f'Rate wait: {result['rate_limit_wait']:.2f}s')
    print(f'\n{'=' * 60}\n  DEMO COMPLETE\n{'=' * 60}')
    print('\nQuery log summary:')
    df = rag.show_logs()
    if df is not None:
        print(df[['query_id', 'latency_total_secs', 'tokens_total', 'chunks_used_in_llm', 'rate_limit_wait_secs']].to_string())
