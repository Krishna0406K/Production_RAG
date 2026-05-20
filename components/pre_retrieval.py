import re
import hashlib
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
for resource in ('tokenizers/punkt', 'tokenizers/punkt_tab'):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[1], quiet=True)

@dataclass
class Chunk:
    chunk_id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class SlidingWindowChunker:

    def __init__(self, chunk_size: int=5, overlap: int=1, min_chunk_len: int=40):
        if overlap >= chunk_size:
            raise ValueError('overlap must be less than chunk_size')
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_len = min_chunk_len

    def _extract_headings(self, text: str) -> List[str]:
        headings = []
        for line in text.splitlines():
            s = line.strip()
            if s.startswith('#'):
                headings.append(s.lstrip('#').strip())
            elif re.match('^[A-Z][A-Z\\s\\-]{3,60}$', s):
                headings.append(s)
        return headings

    def chunk(self, text: str, source: str='unknown') -> List['Chunk']:
        sentences = sent_tokenize(text)
        headings = self._extract_headings(text)
        chunks = []
        stride = self.chunk_size - self.overlap
        for i in range(0, len(sentences), stride):
            window = sentences[i:i + self.chunk_size]
            content = ' '.join(window).strip()
            if len(content) < self.min_chunk_len:
                continue
            chunk_id = hashlib.md5(content.encode()).hexdigest()[:12]
            chunks.append(Chunk(chunk_id=chunk_id, content=content, metadata={'source': source, 'chunk_index': len(chunks), 'chunk_level': 'paragraph', 'token_count': len(content.split()), 'headings': headings[:3], 'parent_id': None, 'children_ids': [], 'created_at': datetime.utcnow().isoformat()}))
        print(f"[SlidingWindowChunker] '{source}' → {len(chunks)} chunks (size={self.chunk_size}, overlap={self.overlap})")
        return chunks

class AdvancedEmbedder:
    MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

    def __init__(self, batch_size: int=32):
        print(f"[AdvancedEmbedder] Loading '{self.MODEL_NAME}' …")
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.batch_size = batch_size
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f'[AdvancedEmbedder] Ready — dim={self.dim}')

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=len(texts) > 100, convert_to_numpy=True, normalize_embeddings=True)

    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        if not chunks:
            return chunks
        vectors = self.embed_texts([c.content for c in chunks])
        for chunk, vec in zip(chunks, vectors):
            chunk.embedding = vec
        print(f'[AdvancedEmbedder] Embedded {len(chunks)} chunks.')
        return chunks

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True)

class HierarchicalIndexer:

    def __init__(self, embedder: AdvancedEmbedder, fine_chunk_size: int=2, coarse_chunk_size: int=6):
        self.embedder = embedder
        self.fine_chunker = SlidingWindowChunker(chunk_size=fine_chunk_size, overlap=0)
        self.coarse_chunker = SlidingWindowChunker(chunk_size=coarse_chunk_size, overlap=1)

    def build(self, text: str, source: str='unknown') -> Tuple[List[Chunk], List[Chunk]]:
        fine_chunks = self.fine_chunker.chunk(text, source=source)
        coarse_chunks = self.coarse_chunker.chunk(text, source=source)
        for c in fine_chunks:
            c.metadata['chunk_level'] = 'fine'
        for c in coarse_chunks:
            c.metadata['chunk_level'] = 'coarse'
        for fine in fine_chunks:
            for coarse in coarse_chunks:
                if fine.content[:60] in coarse.content:
                    fine.metadata['parent_id'] = coarse.chunk_id
                    if fine.chunk_id not in coarse.metadata['children_ids']:
                        coarse.metadata['children_ids'].append(fine.chunk_id)
                    break
        self.embedder.embed_chunks(fine_chunks)
        self.embedder.embed_chunks(coarse_chunks)
        print(f"[HierarchicalIndexer] Built {len(fine_chunks)} fine + {len(coarse_chunks)} coarse chunks for '{source}'.")
        return (fine_chunks, coarse_chunks)

class MetadataAttacher:
    _STOPWORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'this', 'that', 'these', 'those', 'it', 'its', 'as', 'not', 'also', 'they', 'them', 'their', 'we', 'our', 'you', 'your', 'he', 'she', 'his', 'her', 'which', 'who', 'what', 'when', 'where', 'how', 'all', 'more'}

    def _detect_language(self, text: str) -> str:
        common_en = {'the', 'is', 'are', 'and', 'of', 'to', 'in', 'a'}
        return 'en' if common_en & set(text.lower().split()) else 'unknown'

    def _extract_keywords(self, text: str, top_n: int=5) -> List[str]:
        words = re.findall('\\b[a-z]{4,}\\b', text.lower())
        freq: Dict[str, int] = {}
        for w in words:
            if w not in self._STOPWORDS:
                freq[w] = freq.get(w, 0) + 1
        return sorted(freq, key=lambda k: freq[k], reverse=True)[:top_n]

    def _extract_entities(self, text: str) -> List[str]:
        pattern = '\\b([A-Z][a-z]+(?:\\s[A-Z][a-z]+){0,2})\\b'
        return list(set(re.findall(pattern, text)))[:8]

    def attach(self, chunks: List[Chunk]) -> List[Chunk]:
        for chunk in chunks:
            chunk.metadata.update({'language': self._detect_language(chunk.content), 'keywords': self._extract_keywords(chunk.content), 'entities': self._extract_entities(chunk.content)})
        print(f'[MetadataAttacher] Attached metadata to {len(chunks)} chunks.')
        return chunks

class PreRetrievalPipeline:

    def __init__(self):
        self.embedder = AdvancedEmbedder()
        self.chunker = SlidingWindowChunker(chunk_size=5, overlap=1)
        self.hierarchical = HierarchicalIndexer(self.embedder)
        self.meta_attach = MetadataAttacher()

    def run(self, text: str, source: str='document', use_hierarchical: bool=True) -> Dict[str, List[Chunk]]:
        print('\n' + '═' * 60)
        print('  PHASE 1 — PRE-RETRIEVAL & DATA INDEXING')
        print('═' * 60)
        flat_chunks = self.chunker.chunk(text, source=source)
        self.embedder.embed_chunks(flat_chunks)
        self.meta_attach.attach(flat_chunks)
        result: Dict[str, List[Chunk]] = {'flat': flat_chunks, 'fine': [], 'coarse': []}
        if use_hierarchical:
            fine, coarse = self.hierarchical.build(text, source=source)
            self.meta_attach.attach(fine)
            self.meta_attach.attach(coarse)
            result['fine'] = fine
            result['coarse'] = coarse
        print(f'\n[Phase 1 Complete] flat={len(result['flat'])} | fine={len(result['fine'])} | coarse={len(result['coarse'])}\n')
        return result
