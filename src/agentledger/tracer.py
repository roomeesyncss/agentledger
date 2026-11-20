"""Tracer: span creation, context propagation, and the ``@trace`` decorator.

Parent/child links propagate through :mod:`contextvars`, so nesting works across
``await`` boundaries and inside ``asyncio`` tasks without passing spans around.
"""

from __future__ import annotations

import functools
import inspect
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar, overload

from .pricing import DEFAULT_PRICING, Pricing, price_for
from .span import Span, SpanKind

F = TypeVar("F", bound=Callable[..., Any])

_current: ContextVar[Span | None] = ContextVar("agentledger_current_span", default=None)


def current_span() -> Span | None:
    """The innermost active span in this context, if any."""
    return _current.get()


class Tracer:
    """Creates spans and hands finished ones to exporters.

    Args:
        exporters: Objects with an ``export(span)`` method (see :mod:`agentledger.exporters`).
        pricing: Extra or overriding model prices, merged over the defaults.
        capture_io: Record function arguments and return values on ``@trace`` spans.
        max_io_chars: Truncate captured strings and reprs to this length.
    """

    def __init__(
        self,
        exporters: Iterable[Any] = (),
        *,
        pricing: Mapping[str, Pricing] | None = None,
        capture_io: bool = True,
        max_io_chars: int = 2000,
    ) -> None:
        self.exporters = list(exporters)
        self.pricing: dict[str, Pricing] = {**DEFAULT_PRICING, **(pricing or {})}
        self.capture_io = capture_io
        self.max_io_chars = max_io_chars
        self._lock = threading.Lock()
        self._finished: list[Span] = []

    @property
    def spans(self) -> list[Span]:
        """All spans finished by this tracer, in completion order."""
        with self._lock:
            return list(self._finished)

    def reset(self) -> None:
        with self._lock:
            self._finished.clear()

    @contextmanager
    def span(
        self, name: str, kind: SpanKind = "chain", *, input: Any = None, **attributes: Any
    ) -> Iterator[Span]:
        """Time a block of work as a child of the current span."""
        parent = _current.get()
        s = Span(
            name=name,
            kind=kind,
            trace_id=parent.trace_id if parent else uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
            parent_id=parent.span_id if parent else None,
            start_time=time.time(),
            attributes=dict(attributes),
            input=self.safe(input) if input is not None else None,
        )
        token = _current.set(s)
        t0 = time.perf_counter()
        try:
            yield s
        except BaseException as exc:
            s.status = "error"
            s.error = f"{type(exc).__name__}: {exc}"[: self.max_io_chars]
            raise
        finally:
            _current.reset(token)
            s.end_time = s.start_time + (time.perf_counter() - t0)
            self._finish(s)

    def _finish(self, span: Span) -> None:
        if span.usage and span.usage.cost_usd is None:
            p = price_for(span.usage.model, self.pricing)
            if p is not None:
                span.usage.cost_usd = round(p.cost(span.usage), 8)
        if span.output is not None:
            span.output = self.safe(span.output)
        with self._lock:
            self._finished.append(span)
        for exporter in self.exporters:
            try:
                exporter.export(span)
            except Exception as exc:  # an exporter must never break the agent
                import warnings

                warnings.warn(f"agentledger exporter {exporter!r} failed: {exc}", stacklevel=2)

    @overload
    def trace(self, fn: F, /) -> F: ...
    @overload
    def trace(
        self, name: str | None = None, *, kind: SpanKind = "chain", capture_io: bool | None = None
    ) -> Callable[[F], F]: ...

    def trace(self, name: Any = None, *, kind: SpanKind = "chain", capture_io: bool | None = None):
        """Decorate a sync or async function so each call becomes a span.

        Usable bare (``@tracer.trace``) or configured
        (``@tracer.trace("search", kind="tool")``).
        """
        return _decorator(lambda: self, name, kind, capture_io)

    # -- safe serialisation ----------------------------------------------------------

    def safe(self, value: Any, _depth: int = 0) -> Any:
        """Convert ``value`` to bounded, JSON-serialisable data."""
        limit = self.max_io_chars
        if value is None or isinstance(value, bool | int | float):
            return value
        if isinstance(value, str):
            return value if len(value) <= limit else value[:limit] + f"…(+{len(value) - limit})"
        if _depth >= 4:
            return self.safe(repr(value), _depth)
        if isinstance(value, Mapping):
            items = list(value.items())[:50]
            return {str(k): self.safe(v, _depth + 1) for k, v in items}
        if isinstance(value, list | tuple | set | frozenset):
            return [self.safe(v, _depth + 1) for v in list(value)[:50]]
        if hasattr(value, "model_dump"):  # pydantic / SDK response objects
            try:
                return self.safe(value.model_dump(), _depth + 1)
            except Exception:
                pass
        return self.safe(repr(value), _depth)


def _bind_args(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    try:
        bound = inspect.signature(fn).bind_partial(*args, **kwargs).arguments
    except (TypeError, ValueError):
        return {"args": list(args), "kwargs": kwargs}
    return {k: v for k, v in bound.items() if k not in ("self", "cls")}


def _decorator(
    get_tracer: Callable[[], Tracer], name: Any, kind: SpanKind, capture_io: bool | None
) -> Any:
    def wrap(fn: F) -> F:
        span_name = name if isinstance(name, str) else fn.__qualname__

        def capture(t: Tracer) -> bool:
            return t.capture_io if capture_io is None else capture_io

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                t = get_tracer()
                inp = _bind_args(fn, args, kwargs) if capture(t) else None
                with t.span(span_name, kind, input=inp) as s:
                    result = await fn(*args, **kwargs)
                    if capture(t):
                        s.output = result
                    return result

            return awrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t = get_tracer()
            inp = _bind_args(fn, args, kwargs) if capture(t) else None
            with t.span(span_name, kind, input=inp) as s:
                result = fn(*args, **kwargs)
                if capture(t):
                    s.output = result
                return result

        return wrapper  # type: ignore[return-value]

    if callable(name) and not isinstance(name, str):  # used bare: @trace
        fn, name = name, None
        return wrap(fn)
    return wrap


# -- module-level default tracer -------------------------------------------------------

_default_lock = threading.Lock()
_default: Tracer = Tracer()


def get_tracer() -> Tracer:
    return _default


def configure(*exporters: Any, **kwargs: Any) -> Tracer:
    """Replace the global tracer used by :func:`span` and :func:`trace`."""
    global _default
    with _default_lock:
        _default = Tracer(exporters, **kwargs)
    return _default


def span(name: str, kind: SpanKind = "chain", *, input: Any = None, **attributes: Any) -> Any:
    """Context manager on the global tracer: ``with agentledger.span("plan"): ...``."""
    return get_tracer().span(name, kind, input=input, **attributes)


def trace(name: Any = None, *, kind: SpanKind = "chain", capture_io: bool | None = None) -> Any:
    """Decorator on the global tracer, resolved at call time so ``configure`` can run later."""
    return _decorator(get_tracer, name, kind, capture_io)
