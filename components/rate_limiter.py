import time
import threading
import logging
from typing import Optional
logger = logging.getLogger('rag.rate_limiter')

class TokenBucket:

    def __init__(self, capacity: float=6000.0, refill_rate: float=100.0):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        tokens_to_add = elapsed * self.refill_rate
        self._tokens = min(self.capacity, self._tokens + tokens_to_add)
        self._last_refill = now

    def consume(self, tokens: float) -> float:
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                logger.debug(f'[RateLimiter] Consumed {tokens:.0f} tokens. Remaining: {self._tokens:.0f}')
                return 0.0
            tokens_short = tokens - self._tokens
            wait_seconds = tokens_short / self.refill_rate
            logger.info(f'[RateLimiter] Bucket low ({self._tokens:.0f} tokens). Need {tokens:.0f}. Sleeping {wait_seconds:.2f}s …')
        time.sleep(wait_seconds)
        with self._lock:
            self._refill()
            self._tokens -= tokens
        return wait_seconds

    @property
    def available(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens
GROQ_CALL_COSTS = {'query_rewrite': 120, 'query_expand': 280, 'compress_chunk': 400, 'cot_generation': 800, 'direct_llm': 400}

class GroqRateLimiter:

    def __init__(self, tokens_per_minute: int=6000):
        self.tpm = tokens_per_minute
        self.bucket = TokenBucket(capacity=float(tokens_per_minute), refill_rate=tokens_per_minute / 60.0)
        self._total_waited = 0.0
        self._total_calls = 0
        self._total_tokens = 0.0
        logger.info(f'[GroqRateLimiter] Initialised. Limit: {tokens_per_minute} TPM ({tokens_per_minute / 60:.1f} tokens/sec)')

    def wait(self, call_type: str, custom_tokens: Optional[int]=None) -> float:
        if call_type not in GROQ_CALL_COSTS and custom_tokens is None:
            raise ValueError(f"Unknown call_type '{call_type}'. Valid options: {list(GROQ_CALL_COSTS.keys())} or pass custom_tokens=N.")
        tokens = custom_tokens if custom_tokens is not None else GROQ_CALL_COSTS[call_type]
        waited = self.bucket.consume(tokens)
        self._total_waited += waited
        self._total_calls += 1
        self._total_tokens += tokens
        if waited > 0:
            print(f"  [RateLimit] Waited {waited:.2f}s before '{call_type}' ({tokens} tokens estimated)")
        return waited

    def status(self) -> dict:
        return {'tokens_available': round(self.bucket.available, 1), 'tokens_capacity': self.tpm, 'total_calls_gated': self._total_calls, 'total_tokens_spent': round(self._total_tokens, 0), 'total_wait_secs': round(self._total_waited, 3)}

    def reset_stats(self):
        self._total_waited = 0.0
        self._total_calls = 0
        self._total_tokens = 0.0
