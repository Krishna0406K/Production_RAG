import time
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
logger = logging.getLogger('rag.query')

@dataclass
class QueryLog:
    query_id: str = ''
    query_text: str = ''
    timestamp: str = ''
    t_phase2_start: Optional[float] = None
    t_phase2_end: Optional[float] = None
    t_phase3_start: Optional[float] = None
    t_phase3_end: Optional[float] = None
    t_phase4_start: Optional[float] = None
    t_phase4_end: Optional[float] = None
    t_total_start: Optional[float] = None
    t_total_end: Optional[float] = None
    chunks_retrieved: int = 0
    chunks_after_rerank: int = 0
    chunks_after_compress: int = 0
    chunks_used_in_llm: int = 0
    tokens_query_rewrite: int = 0
    tokens_compress_total: int = 0
    tokens_generation: int = 0
    tokens_total: int = 0
    rate_limit_wait_secs: float = 0.0
    strategy: str = ''
    fallback_used: bool = False
    error: Optional[str] = None

    def phase2_latency(self) -> Optional[float]:
        if self.t_phase2_start and self.t_phase2_end:
            return round(self.t_phase2_end - self.t_phase2_start, 3)
        return None

    def phase3_latency(self) -> Optional[float]:
        if self.t_phase3_start and self.t_phase3_end:
            return round(self.t_phase3_end - self.t_phase3_start, 3)
        return None

    def phase4_latency(self) -> Optional[float]:
        if self.t_phase4_start and self.t_phase4_end:
            return round(self.t_phase4_end - self.t_phase4_start, 3)
        return None

    def total_latency(self) -> Optional[float]:
        if self.t_total_start and self.t_total_end:
            return round(self.t_total_end - self.t_total_start, 3)
        return None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop('t_phase2_start')
        d.pop('t_phase2_end')
        d.pop('t_phase3_start')
        d.pop('t_phase3_end')
        d.pop('t_phase4_start')
        d.pop('t_phase4_end')
        d.pop('t_total_start')
        d.pop('t_total_end')
        d['latency_phase2_secs'] = self.phase2_latency()
        d['latency_phase3_secs'] = self.phase3_latency()
        d['latency_phase4_secs'] = self.phase4_latency()
        d['latency_total_secs'] = self.total_latency()
        return d

class LatencyLogger:

    def __init__(self, log_dir: str='logs', log_file: str='queries.jsonl', verbose: bool=True):
        self.verbose = verbose
        self.log_path = Path(log_dir) / log_file
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"[LatencyLogger] Logging queries to '{self.log_path}'")

    def start_query(self, query_text: str) -> QueryLog:
        import hashlib
        now = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()
        query_id = hashlib.md5((query_text + timestamp).encode()).hexdigest()[:8]
        log = QueryLog()
        log.query_id = query_id
        log.query_text = query_text[:120]
        log.timestamp = timestamp
        log.t_total_start = now
        return log

    def finish(self, log: QueryLog, answer: str='') -> Dict[str, Any]:
        log.t_total_end = time.perf_counter()
        log.tokens_total = log.tokens_query_rewrite + log.tokens_compress_total + log.tokens_generation
        d = log.to_dict()
        d['answer_preview'] = answer[:100] if answer else ''
        self._write_jsonl(d)
        if self.verbose:
            self._print_summary(log)
        return d

    def _write_jsonl(self, record: Dict[str, Any]):
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, default=str) + '\n')
        except Exception as e:
            logger.warning(f'[LatencyLogger] Could not write to log: {e}')

    def _print_summary(self, log: QueryLog):
        sep = '─' * 56
        total = log.total_latency()
        p2 = log.phase2_latency()
        p3 = log.phase3_latency()
        p4 = log.phase4_latency()
        bar = ''
        if total and p2 and p3 and p4:
            bar = _latency_bar(p2, p3, p4, total, width=40)
        lines = [f'\n┌─ Query log [{log.query_id}] {'─' * 30}', f'│  Query    : {log.query_text[:55]}', f'│  Total    : {_fmt_ms(total)}', f'│  Phase 2  : {_fmt_ms(p2)}  (retrieval)', f'│  Phase 3  : {_fmt_ms(p3)}  (rerank + compress)', f'│  Phase 4  : {_fmt_ms(p4)}  (generation)']
        if bar:
            lines.append(f'│  Time bar : {bar}')
        lines += [f'│  Chunks   : {log.chunks_retrieved} → {log.chunks_after_rerank} → {log.chunks_after_compress} → {log.chunks_used_in_llm}', f'│  Tokens   : rewrite={log.tokens_query_rewrite} compress={log.tokens_compress_total} gen={log.tokens_generation} total={log.tokens_total}']
        if log.rate_limit_wait_secs > 0:
            lines.append(f'│  Waited   : {log.rate_limit_wait_secs:.2f}s (rate limit)')
        if log.fallback_used:
            lines.append('│  Fallback : ⚠ compression filtered all chunks')
        if log.error:
            lines.append(f'│  Error    : {log.error}')
        lines.append('└' + '─' * 55)
        print('\n'.join(lines))

def _fmt_ms(seconds: Optional[float]) -> str:
    if seconds is None:
        return 'n/a'
    if seconds < 1.0:
        return f'{seconds * 1000:.0f}ms'
    return f'{seconds:.2f}s'

def _latency_bar(p2: float, p3: float, p4: float, total: float, width: int=40) -> str:
    if total == 0:
        return ''
    c2 = max(1, round(p2 / total * width))
    c3 = max(1, round(p3 / total * width))
    c4 = max(1, width - c2 - c3)
    bar = '\x1b[34m' + '█' * c2 + '\x1b[33m' + '█' * c3 + '\x1b[35m' + '█' * c4 + '\x1b[0m'
    label = '  P2=blue P3=amber P4=purple'
    return f'[{bar}]{label}'

def load_logs(log_path: str='logs/queries.jsonl'):
    try:
        import pandas as pd
        df = pd.read_json(log_path, lines=True)
        print(f"[LatencyLogger] Loaded {len(df)} query logs from '{log_path}'")
        return df
    except ImportError:
        print('pandas not installed. Run: pip install pandas')
        return None
    except Exception as e:
        print(f'[LatencyLogger] Could not load logs: {e}')
        return None
