from typing import List, Dict, Any
from components.pre_retrieval import Chunk

def _build_context(chunks: List[Chunk], max_words: int=1800) -> str:
    parts = []
    word_count = 0
    for i, chunk in enumerate(chunks, start=1):
        words = chunk.content.split()
        if word_count + len(words) > max_words:
            break
        parts.append(f'[{i}] {chunk.content}')
        word_count += len(words)
    return '\n\n'.join(parts)

class ChainOfThoughtGenerator:
    _SYSTEM_PROMPT = 'You are a careful, analytical assistant. You always reason step-by-step using the provided context before giving your final answer. You never introduce facts that are not present in the context.'
    _COT_TEMPLATE = 'Use the following context passages to answer the question.\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\n\nThink through this step by step:\nStep 1 — Which passages from the context are directly relevant to the question? List them by number.\nStep 2 — What do those passages tell us? Reason through the key facts and how they connect.\nStep 3 — Final Answer: Write a clear, concise answer based only on the context above.'

    def __init__(self, groq_client, model: str='openai/gpt-oss-120b', max_tokens: int=600, temperature: float=0.2):
        self.client = groq_client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, query: str, chunks: List[Chunk]) -> str:
        context = _build_context(chunks)
        prompt = self._COT_TEMPLATE.format(context=context, query=query)
        resp = self.client.chat.completions.create(model=self.model, messages=[{'role': 'system', 'content': self._SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}], max_tokens=self.max_tokens, temperature=self.temperature)
        answer = resp.choices[0].message.content.strip()
        tokens = resp.usage.total_tokens
        print(f'[CoT] Answer generated. Tokens used: {tokens}')
        return answer

class GenerationPipeline:

    def __init__(self, groq_client):
        self.cot = ChainOfThoughtGenerator(groq_client)

    def generate(self, query: str, chunks: List[Chunk]) -> Dict[str, Any]:
        print('\n' + '═' * 60)
        print('  PHASE 4 — GENERATION (Chain-of-Thought)')
        print('═' * 60)
        answer = self.cot.generate(query, chunks)
        print(f'\n[Phase 4 Complete] Answer: {len(answer)} chars.\n')
        return {'answer': answer, 'strategy': 'chain_of_thought'}
