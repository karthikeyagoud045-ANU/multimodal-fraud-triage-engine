import asyncio
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar


F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


@dataclass
class TokenTracker:
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_latency_seconds: float = 0.0
    call_count: int = 0
    calls_by_name: Dict[str, int] = field(default_factory=dict)

    def record(
        self,
        name: str,
        latency_seconds: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        self.call_count += 1
        self.calls_by_name[name] = self.calls_by_name.get(name, 0) + 1
        self.total_latency_seconds += latency_seconds
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def average_latency_seconds(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_latency_seconds / self.call_count

    def estimated_input_cost_usd(self) -> float:
        mini_calls = self.calls_by_name.get("text_extractor", 0)
        vlm_calls = self.calls_by_name.get("vlm_inspector", 0)
        total_calls = mini_calls + vlm_calls
        if total_calls == 0 or self.total_prompt_tokens == 0:
            return 0.0
        mini_share = mini_calls / total_calls
        vlm_share = vlm_calls / total_calls
        return (self.total_prompt_tokens * mini_share * 0.15 / 1_000_000) + (
            self.total_prompt_tokens * vlm_share * 2.50 / 1_000_000
        )

    def as_dict(self) -> dict:
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_input_cost_usd": round(self.estimated_input_cost_usd(), 6),
            "average_latency_seconds": round(self.average_latency_seconds, 3),
            "call_count": self.call_count,
            "calls_by_name": self.calls_by_name,
        }


GLOBAL_TRACKER = TokenTracker()


def track_latency(name: str, tracker: Optional[TokenTracker] = None) -> Callable[[F], F]:
    active_tracker = tracker or GLOBAL_TRACKER

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            active_tracker.record(name=name, latency_seconds=time.perf_counter() - start)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


async def gather_with_concurrency(limit: int, *tasks: Awaitable[Any]) -> list:
    semaphore = asyncio.Semaphore(limit)

    async def run(task: Awaitable[Any]) -> Any:
        async with semaphore:
            return await task

    return await asyncio.gather(*(run(task) for task in tasks))

