"""Settings once, override per call: ``configure()`` and ``context()``.

Where a call goes is decided in this order, first match wins:

1. the keyword on the call itself (``simulate(agent=...)``, ``Judge(model=...)``);
2. the innermost ``with wai.context(...)`` block;
3. what ``wai.configure(...)`` set for the process;
4. the environment (``WHILEAI_AGENT``, ``OPENAI_API_KEY``, ...);
5. the package default: the model While hosts, on the key ``whileai login`` saved.

    import whileai as wai

    wai.configure(
        agent=wai.OpenAI("gpt-4.1-mini", api_key="sk-..."),
        judge=wai.Anthropic("claude-haiku-4-5", api_key="sk-ant-..."),
    )
    print(wai.settings)   # says which model and which key each role uses

A key given on a backend object is kept for that provider, so every call
to that provider in the process finds it. ``api_key=`` on ``configure``
is the While account key, for the hosted models and the platform.

This module imports nothing from the engine so the engine can import it.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any

ROLES = ("agent", "judge", "simulator")


def spec_of(value: Any) -> Any:
    """The backend spec string for ``value``: a backend object's ``.spec``,
    a string as given, a callable as given, ``None`` for "the default"."""
    if value is None or isinstance(value, str) or callable(value):
        return value
    spec = getattr(value, "spec", None)
    if spec is None or isinstance(spec, str):
        return spec
    raise TypeError(f"not a backend: {value!r}")


@dataclass
class Settings:
    """The process-wide answers to "which model" and "which key"."""

    agent: str | None = None
    judge: str | None = None
    simulator: str | None = None
    #: the While account key (hosted models, platform); ``whileai login`` sets it too
    api_key: str | None = None
    #: provider -> key, filled from backend objects: ``openai``, ``anthropic``,
    #: ``vllm``, ``typesafe``. ``ollama`` never needs one.
    keys: dict[str, str] = field(default_factory=dict)

    def key_for(self, provider: str) -> str | None:
        return self.keys.get(provider)

    def __repr__(self) -> str:
        def show(role: str) -> str:
            value = getattr(self, role)
            return value if value else "default (While hosted)"

        parts = [f"{role}={show(role)}" for role in ROLES]
        parts.append(f"api_key={'set' if self.api_key else 'unset'}")
        if self.keys:
            parts.append("keys=" + ",".join(sorted(self.keys)))
        return "Settings(" + ", ".join(parts) + ")"


_base = Settings()
_local = threading.local()


def _stack() -> list[Settings]:
    stack = getattr(_local, "stack", None)
    if stack is None:
        stack = _local.stack = []
    return stack


def current() -> Settings:
    """The settings in force: the innermost ``context()`` or the process base."""
    stack = _stack()
    return stack[-1] if stack else _base


def _absorb(target: Settings, role: str, value: Any) -> None:
    """Record a backend (or spec string) under ``role``, and its key if it has one."""
    provider = getattr(value, "provider", None)
    key = getattr(value, "api_key", None)
    if provider and key:
        target.keys[str(provider)] = str(key)
    setattr(target, role, spec_of(value))


def configure(
    *,
    agent: Any = None,
    judge: Any = None,
    simulator: Any = None,
    api_key: str | None = None,
) -> Settings:
    """Set the process defaults. Each argument is a backend object
    (``wai.OpenAI(...)``), a spec string (``"openai:gpt-4.1-mini"``) or
    ``None`` to leave that role as it is. Returns the settings, whose repr
    says what each role resolves to.

    * ``agent``: the policy under test.
    * ``judge``: the grader; never the same model as the agent by default.
    * ``simulator``: the writer of user messages; the agent's model unless set.
    * ``api_key``: the While account key.
    """
    if agent is not None:
        _absorb(_base, "agent", agent)
    if judge is not None:
        _absorb(_base, "judge", judge)
    if simulator is not None:
        _absorb(_base, "simulator", simulator)
    if api_key is not None:
        _base.api_key = str(api_key).strip() or None
    return _base


def reset() -> Settings:
    """Forget everything ``configure()`` set. Tests call this."""
    global _base
    _base = Settings()
    _stack().clear()
    return _base


@contextlib.contextmanager
def context(
    *,
    agent: Any = None,
    judge: Any = None,
    simulator: Any = None,
    api_key: str | None = None,
) -> Iterator[Settings]:
    """Override the settings inside a ``with`` block, on this thread only.

    with wai.context(judge=wai.OpenAI("gpt-4.1")):
        strict = data.grade(wai.Judge(rubric=RUBRIC))
    """
    scoped = replace(current(), keys=dict(current().keys))
    if agent is not None:
        _absorb(scoped, "agent", agent)
    if judge is not None:
        _absorb(scoped, "judge", judge)
    if simulator is not None:
        _absorb(scoped, "simulator", simulator)
    if api_key is not None:
        scoped.api_key = str(api_key).strip() or None
    stack = _stack()
    stack.append(scoped)
    try:
        yield scoped
    finally:
        stack.pop()


class _SettingsProxy:
    """``wai.settings``: always the settings in force, never a stale copy."""

    def __getattr__(self, name: str) -> Any:
        return getattr(current(), name)

    def __repr__(self) -> str:
        return repr(current())


settings = _SettingsProxy()

__all__ = ["Settings", "configure", "context", "current", "reset", "settings", "spec_of"]
