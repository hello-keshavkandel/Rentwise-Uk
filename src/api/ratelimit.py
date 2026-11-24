"""
ratelimit.py  —  per-API-key rate limiting (sliding window).

In-process and dependency-free: correct for a single API instance. With several
replicas, swap the store for Redis (same interface) — noted in DEPLOYMENT.md.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class SlidingWindowLimiter:
    def __init__(self, limit: int = 60, window_s: float = 60.0, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window_s, clock
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int, float]:
        """Record a hit for `key`. Returns (allowed, remaining, retry_after_seconds)."""
        now = self.clock()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False, 0, max(0.0, self.window - (now - q[0]))
            q.append(now)
            return True, self.limit - len(q), 0.0
