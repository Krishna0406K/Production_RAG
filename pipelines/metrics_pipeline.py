import sys
from pathlib import Path
import os
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from groq import Groq
from components.pre_retrieval import PreRetrievalPipeline, Chunk, AdvancedEmbedder
from components.retrieval import RetrievalPipeline, VectorStore
from components.post_retrieval import PostRetrievalPipeline
from components.generation import GenerationPipeline
from components.document_loader import load_document, load_documents
from datasets import Dataset
import pandas as pd
from ragas.llms import LangchainLLMWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas import evaluate
from langchain_groq import ChatGroq
Groq_api_key = os.getenv('GROQ_API_KEY')

def _direct_llm(groq_client, query: str) -> str:
    resp = groq_client.chat.completions.create(model='openai/gpt-oss-120b', messages=[{'role': 'user', 'content': query}], max_tokens=300, temperature=0.4)
    return resp.choices[0].message.content.strip()

class AdvancedRAGPipeline:

    def __init__(self, groq_api_key: Optional[str]=None, use_compression: bool=True, use_hierarchical: bool=True, reranker_top_k: int=3, retrieval_top_k: int=8):
        api_key = Groq_api_key
        if not api_key:
            raise ValueError("Groq API key is required.\nEither pass groq_api_key='gsk_...' or set GROQ_API_KEY in .env")
        self.groq = Groq(api_key=api_key)
        self.use_compression = use_compression
        self.use_hierarchical = use_hierarchical
        self.retrieval_top_k = retrieval_top_k
        print('\n▶ Initialising Advanced RAG Pipeline …')
        self.pre_retrieval = PreRetrievalPipeline()
        self.embedder = self.pre_retrieval.embedder
        dim = self.embedder.dim
        self.flat_store = VectorStore(dim=dim)
        self.fine_store = VectorStore(dim=dim)
        self.coarse_store = VectorStore(dim=dim)
        self._retrieval: Optional[RetrievalPipeline] = None
        self._post: Optional[PostRetrievalPipeline] = None
        self._generation: Optional[GenerationPipeline] = None
        self._reranker_top_k = reranker_top_k
        print('▶ Pipeline ready. Call .ingest() to add documents.\n')

    def ingest(self, source: str) -> Dict[str, int]:
        doc = load_document(source)
        text = doc['text']
        source_name = doc['source']
        print(f"   INGESTING: '{source_name}'")
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
        return [self.ingest(src) for src in sources]

    def _build(self):
        self._retrieval = RetrievalPipeline(flat_store=self.flat_store, fine_store=self.fine_store, coarse_store=self.coarse_store, embedder=self.embedder, groq_client=self.groq, top_k=self.retrieval_top_k)
        self._post = PostRetrievalPipeline(groq_client=self.groq, reranker_top_k=self._reranker_top_k, use_compression=self.use_compression)
        self._generation = GenerationPipeline(groq_client=self.groq)

    def query(self, query: str) -> Dict[str, Any]:
        if not self._retrieval:
            raise RuntimeError('No documents ingested. Call .ingest(source) first.')
        print(f'║   QUERY: {query[:50]:<50}║')
        retrieved_chunks, retrieval_meta = self._retrieval.retrieve(query)
        if not retrieved_chunks:
            print('[Pipeline] No chunks retrieved. Falling back to direct LLM.')
            answer = _direct_llm(self.groq, query)
            return {'answer': answer, 'strategy': 'direct_llm_fallback', 'chunks_used': 0, 'retrieval_meta': retrieval_meta, 'post_meta': {}, 'raw_context_strings': []}
        final_chunks, post_meta = self._post.run(query, retrieved_chunks)
        if not final_chunks:
            print('[Pipeline] All chunks filtered. Using pre-compression chunks.')
            final_chunks = retrieved_chunks[:2]
        gen_result = self._generation.generate(query, final_chunks)
        raw_context_strings = [chunk.content for chunk in final_chunks]
        return {'answer': gen_result['answer'], 'strategy': gen_result['strategy'], 'chunks_used': len(final_chunks), 'retrieval_meta': retrieval_meta, 'post_meta': post_meta, 'raw_context_strings': raw_context_strings}

    def ask(self, question: str) -> str:
        return self.query(question)['answer']
_FILE_PATH = str('dataset/AI Engineering_ Building Applications With Foundation Models by Chip Huyen (1).pdf')
_DEMO_QUESTIONS = ['What is the idea behind the context retrieval?', 'What are the four primary cognitive biases associated with using an LLM as an AI-as-a-Judge framework?']
_GROUND_TRUTH_ANSWERS = ['The idea behind contextual retrieval is to augment each chunk with relevant context to make it easier to retrieve the relevant chunks. A simple technique is to augment a chunk with metadata like tags and keywords. For ecommerce, a product can be augmented by its description and reviews. Images and videos can be queried by their titles or captions.', 'The four primary biases are: 1. Inconsistency (probabilistic score variance), 2. Self-Bias (favoring its own outputs, such as GPT-4 giving itself a higher win rate), 3. Position Bias (favoring the first answer displayed in a pairwise comparison), and 4. Verbosity Bias (preferring longer, more detailed responses even if they contain factual errors).']
if __name__ == '__main__':
    print('   ADVANCED RAG PIPELINE — DEMO')
    rag = AdvancedRAGPipeline(use_compression=True, use_hierarchical=True, reranker_top_k=3, retrieval_top_k=8)
    rag.ingest(_FILE_PATH)
    eval_questions = []
    eval_answers = []
    eval_contexts = []
    for i, question in enumerate(_DEMO_QUESTIONS, start=1):
        print(f'\n{'─' * 60}\nQ{i}: {question}\n{'─' * 60}')
        result = rag.query(question)
        print(f'\n📝 ANSWER:\n{result['answer']}')
        eval_questions.append(question)
        eval_answers.append(result['answer'])
        eval_contexts.append(result['raw_context_strings'])
    print('\n' + '═' * 60 + '\n🏁 RUNTIME COMPLETED. INITIALIZING RAGAS EVALUATION...\n' + '═' * 60)
    ragas_dict = {'question': eval_questions, 'answer': eval_answers, 'contexts': eval_contexts, 'ground_truth': _GROUND_TRUTH_ANSWERS}
    eval_dataset = Dataset.from_dict(ragas_dict)
    evaluator_llm = ChatGroq(model='openai/gpt-oss-120b', temperature=0.0)
    ragas_judge_wrapper = LangchainLLMWrapper(evaluator_llm)
    ragas_embed_wrapper = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    metrics_list = [faithfulness, context_precision]
    for metric in metrics_list:
        metric.llm = ragas_judge_wrapper
        if hasattr(metric, 'embeddings'):
            metric.embeddings = ragas_embed_wrapper
    scores = evaluate(dataset=eval_dataset, metrics=metrics_list)
    print('\n📈 RAGAS EVALUATION SUMMARY:')
    print(scores)
    print('\nDetailed breakdown:')
    print(scores.to_pandas()[['faithfulness', 'context_precision']])
