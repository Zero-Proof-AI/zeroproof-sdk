"""OpenAI-compatible chat loop for hosted and local simulation backends."""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import threading
from typing import Any, Callable
from urllib.parse import urlparse

from .diversity import running_turn_mean, sample_turn_budget
from .sandbox import MockEnvironment

DEFAULT_AGENT = (
    "vllm:Qwen/Qwen3-4B-Instruct-2507@"
    "https://zeroproofai--stressd-vllm-serve.modal.run/v1"
)
DEFAULT_SIMULATOR = DEFAULT_AGENT
DEFAULT_HOSTED = DEFAULT_AGENT

_tls = threading.local()


def parse_backend_spec(spec: str) -> tuple[str, str]:
    """Return (base_url, model) for ollama:/vllm:/openai: specs."""
    kind, _, rest = str(spec).partition(":")
    if kind == "ollama":
        return "http://localhost:11434/v1", rest or "llama3.1:8b"
    if kind == "vllm":
        model, _, url = rest.partition("@")
        if not url:
            raise ValueError("vllm spec must be vllm:<model>@<base_url>")
        return url, model
    if kind == "openai":
        return (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
                rest or "gpt-4o-mini")
    raise ValueError(
        f"unsupported backend spec {spec!r}; use ollama:<model>, "
        "vllm:<model>@<url>, or openai:<model>")


def default_agent_spec() -> str:
    """Tool-using rollout model. Hosted Qwen unless ZEROPROOF_AGENT is set."""
    return os.environ.get("ZEROPROOF_AGENT") or DEFAULT_AGENT


def default_simulator_spec() -> str:
    """User-message writer. Same hosted Qwen as the agent unless overridden."""
    return os.environ.get("ZEROPROOF_SURROGATE") or DEFAULT_SIMULATOR


CONTEXT_TOKENS = 4096
_CONTEXT_TOKENS = CONTEXT_TOKENS


def default_max_turns(context_tokens: int | None = None, *,
                      n_tools: int = 0) -> int:
    """Conversation cap. Right tail reaches 40; smaller windows scale down."""
    ctx = int(context_tokens if context_tokens is not None else CONTEXT_TOKENS)
    reserved = 2048 + min(1536, max(0, int(n_tools)) * 64)
    per_turn = 128
    return max(8, min(40, max(1, ctx - reserved) // per_turn))


def touch_hosted(base_url: str | None = None, *, timeout: float = 5.0) -> None:
    """Best-effort GET /v1/models. Keeps the reserved replica awake."""
    key = (os.environ.get("VLLM_API_KEY")
           or os.environ.get("OPENAI_API_KEY") or "")
    url = base_url
    if not url:
        try:
            url, _ = parse_backend_spec(default_agent_spec())
        except ValueError:
            return
    if missing_hosted_key(url, key):
        return
    raw = url if "://" in str(url) else "https://" + str(url)
    parsed = urlparse(raw)
    path = parsed.path.rstrip("/")
    if path.endswith("/models"):
        get_path = path
    elif path.endswith("/v1"):
        get_path = path + "/models"
    else:
        get_path = (path or "") + "/v1/models"
    if not get_path.startswith("/"):
        get_path = "/" + get_path
    headers = {"Connection": "keep-alive"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        if (parsed.scheme or "https") == "https":
            conn = http.client.HTTPSConnection(
                parsed.hostname, parsed.port or 443, timeout=timeout)
        else:
            conn = http.client.HTTPConnection(
                parsed.hostname, parsed.port or 80, timeout=timeout)
        conn.request("GET", get_path, headers=headers)
        conn.getresponse().read()
        conn.close()
    except Exception:
        return


def missing_hosted_key(base_url: str | None = None,
                       api_key: str | None = None) -> str | None:
    """One-sentence auth error for hosted Qwen, or None if a key is present."""
    key = api_key or os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    if key:
        return None
    url = base_url
    if not url:
        try:
            url, _ = parse_backend_spec(default_agent_spec())
        except ValueError:
            return None
    raw = url if "://" in str(url) else "https://" + str(url)
    host = (urlparse(raw).hostname or "").lower()
    if host.endswith("modal.run") or "zeroproof" in host:
        return "Hosted Qwen needs VLLM_API_KEY set in the environment."
    return None


def _estimate_tokens(messages: list[dict], tools: list[dict] | None) -> int:
    blob = json.dumps(messages, default=str, separators=(",", ":"))
    if tools:
        blob += json.dumps(tools, default=str, separators=(",", ":"))
    return max(1, (len(blob) + 2) // 3)


def _last_user_index(messages: list[dict]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return -1


def _shrink_last_user(messages: list[dict], *, frac: float = 0.5) -> bool:
    """Halve a long last user message. Short agent turns stay intact."""
    i = _last_user_index(messages)
    if i < 0:
        return False
    content = str(messages[i].get("content") or "")
    if len(content) < 800:
        return False
    messages[i] = {**messages[i], "content": content[:max(400, int(len(content) * frac))]}
    return True


def complete(base_url: str, model: str, messages: list[dict], *,
             tools: list[dict] | None = None, api_key: str | None = None,
             temperature: float = 0.7, max_tokens: int = 1024,
             timeout: float = 60, n: int = 1) -> dict:
    """POST /chat/completions. Reuses a thread-local keep-alive connection.

    ``n>1`` asks vLLM for several samples on one prefill. The first choice is
    the return value; extra choices are on ``_all`` when the server honors ``n``.
    """
    key = api_key or os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    auth_err = missing_hosted_key(base_url, key)
    if auth_err:
        raise RuntimeError(auth_err)
    raw_url = base_url.rstrip("/")
    if "://" not in raw_url:
        raw_url = "https://" + raw_url
    parsed = urlparse(raw_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        post_path = path
    elif path.endswith("/v1"):
        post_path = path + "/chat/completions"
    else:
        post_path = (path or "") + "/v1/chat/completions"
    if not post_path.startswith("/"):
        post_path = "/" + post_path
    messages = [dict(m) for m in messages]
    room = _CONTEXT_TOKENS - _estimate_tokens(messages, tools) - 64
    while room < 256 and _shrink_last_user(messages):
        room = _CONTEXT_TOKENS - _estimate_tokens(messages, tools) - 64
    want = max(256, min(int(max_tokens), max(256, room)))
    samples = max(1, min(8, int(n)))
    payload: dict[str, Any] = {"model": model, "messages": messages,
                               "temperature": temperature, "max_tokens": want}
    if samples > 1:
        payload["n"] = samples
    if tools:
        payload["tools"] = tools
    headers = {"Content-Type": "application/json", "Connection": "keep-alive"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    conn_key = (parsed.scheme or "https", parsed.hostname, parsed.port, timeout)
    last_err: Exception | None = None
    for _ in range(5):
        payload["messages"] = messages
        body = json.dumps(payload, separators=(",", ":")).encode()
        conn = getattr(_tls, "conn", None)
        if getattr(_tls, "conn_key", None) != conn_key or conn is None:
            if (parsed.scheme or "https") == "https":
                conn = http.client.HTTPSConnection(
                    parsed.hostname, parsed.port or 443, timeout=timeout)
            else:
                conn = http.client.HTTPConnection(
                    parsed.hostname, parsed.port or 80, timeout=timeout)
            _tls.conn, _tls.conn_key = conn, conn_key
        try:
            conn.request("POST", post_path, body=body, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status >= 400:
                err = raw[:400].decode("utf-8", "replace")
                if (resp.status == 400 and "max_tokens" in err
                        and int(payload["max_tokens"]) > 256):
                    payload["max_tokens"] = max(256, int(payload["max_tokens"]) // 2)
                    raise RuntimeError("retry_max_tokens")
                if resp.status == 400 and _shrink_last_user(messages):
                    room = _CONTEXT_TOKENS - _estimate_tokens(messages, tools) - 64
                    payload["max_tokens"] = max(256, min(int(payload["max_tokens"]),
                                                         max(256, room)))
                    raise RuntimeError("retry_shrink_input")
                if resp.status == 400 and payload.get("n"):
                    payload.pop("n", None)
                    raise RuntimeError("retry_drop_n")
                if resp.status in {401, 403}:
                    raise RuntimeError(
                        "Hosted Qwen needs VLLM_API_KEY set in the environment."
                        if not key else
                        f"Hosted Qwen rejected the API key ({resp.status}).")
                if resp.status == 400:
                    raise RuntimeError(
                        f"hosted Qwen rejected the prompt ({resp.status}); "
                        f"it exceeded the {_CONTEXT_TOKENS}-token context.")
                raise RuntimeError(
                    f"{parsed.hostname} returned {resp.status}: {err}")
            choices = json.loads(raw).get("choices") or []
            if not choices:
                raise RuntimeError(f"{parsed.hostname} returned no choices")
            first = dict(choices[0].get("message") or {})
            extras = [dict(c.get("message") or {}) for c in choices]
            if len(extras) > 1:
                first["_all"] = extras
            return first
        except Exception as exc:
            last_err = exc
            try:
                conn.close()
            except Exception:
                pass
            _tls.conn = None
            if str(exc) not in {"retry_max_tokens", "retry_shrink_input",
                                "retry_drop_n"}:
                break
    raise last_err  # type: ignore[misc]


_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def parse_text_tool_calls(text: str) -> list[dict]:
    """Turn Qwen/Hermes ``<tool_call>`` markup into OpenAI tool_calls."""
    if not text or "<tool_call>" not in text:
        return []
    chunks = _TOOL_CALL_BLOCK.findall(text)
    if not chunks:
        chunks = [part.strip() for part in text.split("<tool_call>")[1:] if part.strip()]
    calls = []
    for index, chunk in enumerate(chunks):
        cleaned = re.sub(r",\s*,", ",", chunk.strip())
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            continue
        name = payload.get("name") or payload.get("tool")
        arguments = payload.get("arguments") or payload.get("args") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if not name:
            continue
        calls.append({
            "id": f"call_{index}",
            "type": "function",
            "function": {"name": str(name), "arguments": json.dumps(arguments)},
        })
    return calls


def _strip_tool_markup(text: str) -> str:
    if not text or "<tool_call>" not in text:
        return (text or "").strip()
    cleaned = _TOOL_CALL_BLOCK.sub("", text)
    cleaned = re.sub(r"</?tool_call>", "", cleaned)
    return cleaned.strip()


def _spoken_text(reply: dict) -> str:
    return _strip_tool_markup(str(reply.get("content") or ""))


def _calls_from_reply(reply: dict) -> tuple[list[dict], dict]:
    native = reply.get("tool_calls") or []
    if native:
        return native, reply
    content = reply.get("content") or ""
    parsed = parse_text_tool_calls(content)
    if not parsed:
        return [], reply
    spoken = _strip_tool_markup(content)
    return parsed, {"role": "assistant", "content": spoken or None,
                    "tool_calls": parsed}


def _has_tool_steps(steps: list) -> bool:
    return any(isinstance(s, dict) and s.get("tool") for s in steps)


_PREAMBLE = re.compile(
    r"^(sure|ok|okay|got it|let me|i will|i'll|one (sec|moment)|hang on)\b",
    re.I)
_STOCK_CLOSE = re.compile(
    r"(\s*let me know if you need (anything|any(thing)? else)!?\s*)+$",
    re.I,
)


def _looks_unfinished(text: str, steps: list) -> bool:
    words = (text or "").strip().split()
    if not words:
        return True
    head = " ".join(words[:8])
    return bool(_PREAMBLE.match(head) and len(words) <= 24)


def _last_agent_utterance(steps: list) -> tuple[str, int]:
    """Last spoken agent line, plus the index of the last tool step (-1 if none)."""
    last_tool = -1
    before = ""
    after = ""
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        if step.get("tool"):
            last_tool = i
        text = str(step.get("text") or "")
        if not text:
            continue
        if last_tool >= 0 and i > last_tool:
            after = text
        else:
            before = text
    if last_tool >= 0:
        return after, last_tool
    return before, last_tool


def _finish_on_agent(steps: list, final_text: str) -> dict:
    """Normal rows end on the agent's last utterance, not a dangling user line.

    After tools succeed, ``final_text`` is the post-tool utterance. An earlier
    clarifying question is not reused as the closer.
    """
    while (steps and isinstance(steps[-1], dict)
           and steps[-1].get("user")
           and not steps[-1].get("text")
           and not steps[-1].get("tool")):
        steps.pop()
    spoken, last_tool = _last_agent_utterance(steps)
    passed = str(final_text or "")
    if spoken:
        return {"steps": steps, "final_text": spoken}
    if last_tool < 0:
        return {"steps": steps, "final_text": passed}
    earlier = {str(s.get("text") or "") for s in steps
               if isinstance(s, dict) and str(s.get("text") or "").strip()}
    if passed and passed not in earlier:
        spoken = passed
    return {"steps": steps, "final_text": spoken}


def _agent_asked(text: str) -> bool:
    """True when the agent actually asked the human something."""
    t = str(text or "").strip()
    return bool(t) and ("?" in t)


def _want_followup(message: str, turn_i: int, *,
                   user_turns: int = 1, budget: int = 6,
                   agent_text: str = "") -> bool:
    """True when the human would naturally speak again.

    A question or refusal earns an answer while the turn budget has
    room: the depth cap is ``budget // 2`` user turns (avg_turns=4
    allows 2, avg_turns=8 allows 4). After a completed action about
    half of people react or ask the next thing, when the thread has
    budget for it. Short-budget threads still end on the agent.
    """
    cap = max(2, int(budget) // 2)
    if int(user_turns) >= cap:
        return False
    text = str(agent_text or "")
    if _agent_asked(text):
        return True
    if _AGENT_REFUSAL.search(text):
        return True
    if int(budget) >= 4 and _AGENT_SUCCESS.search(text):
        digest = hashlib.sha256(
            f"{message}:{turn_i}:react".encode()).hexdigest()
        return int(digest[:8], 16) % 2 == 0
    return False


# Follow-up user turns only. Opener temperature lives on the writer (0.45–1.05).
_RESPONSE_TEMP_LO = 0.40
_RESPONSE_TEMP_HI = 0.55
_USER_SIM_SYSTEM = (
    "You write only the human's next spoken line. You are not the assistant. "
    "Ordinary speech. No emojis, no em dashes, no thanks, no you're welcome. "
    "Stay in the same world as the opening line and the tools on this thread. "
    "If the agent asked a question, answer it with a concrete detail a person "
    "here would know (order id, sku, store, size, repo, PR, whatever this "
    "thread is actually about). "
    "Do not invent a git repo, pull request, issue, or branch unless this "
    "thread is already about those. "
    "If they already acted, react: push back, correct them, or ask for the next thing. "
    "Do not acknowledge. Do not repeat their question. Do not describe a persona."
)
_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U0001F600-\U0001F64F"
    "]+"
)
_STOCK_HIT = re.compile(
    r"(i('d| would) be happy to help|of course!|have a great day|"
    r"feel free to (reach out|ask)|let me know if you need|"
    r"i understand your (frustration|concern)|how can i (help|assist) you today)",
    re.I,
)
_MD_MARK = re.compile(r"^#{1,6}\s+|\*\*(.+?)\*\*", re.M)


def _scrub_ai_traces(text: str) -> str:
    """User-side only. Agent voice is left alone."""
    out = _EMOJI.sub("", str(text or ""))
    out = out.replace("\u2014", ", ").replace("\u2013", ", ")
    out = _STOCK_CLOSE.sub("", out)
    out = _STOCK_HIT.sub("", out)
    out = _MD_MARK.sub(lambda m: m.group(1) or "", out)
    return re.sub(r"[ \t]+\n", "\n", re.sub(r" {2,}", " ", out)).strip()


def _render_user_trace(messages: list[dict] | None, steps: list[dict] | None,
                       *, limit: int = 2400) -> str:
    """Plain-text script. Never packed as chat roles."""
    lines: list[str] = []
    if messages:
        for item in messages:
            role = str(item.get("role") or "")
            if role == "system":
                continue
            if role == "user":
                lines.append(f"You said: {item.get('content') or ''}")
                continue
            if role == "assistant":
                spoken = str(item.get("content") or "").strip()
                if spoken:
                    lines.append(f"The agent replied: {spoken}")
                for call in item.get("tool_calls") or []:
                    fn = call.get("function") or {}
                    lines.append(
                        f"The agent used {fn.get('name')}"
                        f"({fn.get('arguments') or '{}'})")
                continue
            if role == "tool":
                name = item.get("name") or "tool"
                lines.append(f"Result ({name}): {str(item.get('content') or '')[:300]}")
    elif steps:
        for step in steps:
            if not isinstance(step, dict):
                continue
            if step.get("user"):
                lines.append(f"You said: {step['user']}")
            if step.get("tool"):
                args = step.get("arguments") or {}
                lines.append(f"The agent used {step.get('tool')}({args})")
                lines.append(f"Result ({step.get('tool')}): {str(step.get('result') or '')[:300]}")
            elif step.get("text"):
                lines.append(f"The agent replied: {step['text']}")
    return "\n".join(lines)[:limit]


_RUBBER_STAMP = re.compile(
    r"^(thanks|thank you|thx|ok|okay|great|perfect|awesome|got it|"
    r"sounds good|cool|you're welcome|you are welcome)[!., ]*$",
    re.I,
)
_THANKS_WORDS = {"thanks", "thank", "thx", "appreciate", "welcome", "glad"}
_ID_FOLLOW = re.compile(
    r"[#/]|\b(pr|issue|repo|branch|sku|store|order|asin|cart)\s*#?\d+"
    r"|\b[\w.-]+/[\w.-]+\b",
    re.I,
)
_CODING_MARK = re.compile(
    r"\b(repo|repository|pull request|\bpr\b|issue\s*#|git branch|"
    r"branch feature/)\b",
    re.I,
)
_CONFIRM_ONLY = re.compile(
    r"^(the )?.{0,40}(successfully|correctly|all good|no issues)\.?$",
    re.I,
)
_AGENT_QUESTION = re.compile(r"\?\s*$|\b(which|what|who|where|can you|could you)\b", re.I)
_AGENT_SUCCESS = re.compile(
    r"\b(done|created|opened|fixed|all set|i (have|'ve|just)|"
    r"successfully|completed)\b",
    re.I,
)
_AGENT_REFUSAL = re.compile(
    r"\b(can't|cannot|won't|unable|not allowed|against (the )?(policy|rules?))\b",
    re.I,
)


def _persona_notes(prior: str) -> str:
    """Backstage typing notes inferred from the opening line. Never copy these."""
    text = str(prior or "")
    notes: list[str] = []
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) < 0.08:
        notes.append("Keep every letter small. Do not mention how you type.")
    elif text[:1].isupper():
        notes.append("Capitalize normally, the way the opening line does.")
    if not re.search(r"[.!?]", text):
        notes.append("Leave out end marks. Never write the word punctuation.")
    elif re.search(r"[.!?]\s*$", text):
        notes.append("End sentences with the usual marks.")
    if re.search(r"\b(again|seriously|come on|still not|ridiculous)\b", text, re.I):
        notes.append("You are still frustrated until this is actually solved.")
    elif re.search(r"\b(asap|right now|waiting)\b", text, re.I):
        notes.append("You are still in a hurry.")
    elif re.search(r"\b(please|thanks|thank you)\b", text, re.I) and len(text) > 40:
        notes.append("Stay polite, but do not just thank them.")
    return " ".join(notes)


def _agent_left_the_ball(agent_text: str) -> bool:
    text = str(agent_text or "")
    if _AGENT_QUESTION.search(text) or text.rstrip().endswith("?"):
        return True
    if _AGENT_SUCCESS.search(text) or _AGENT_REFUSAL.search(text):
        return True
    if text.count("\n-") >= 2 or text.count("\n*") >= 2:
        return True
    return False


def _mostly_thanks(text: str) -> bool:
    words = re.findall(r"[a-z']+", str(text or "").lower())
    if not words:
        return False
    hits = sum(1 for w in words if w in _THANKS_WORDS)
    if hits / len(words) < 0.25:
        return False
    return not re.search(r"\b(also|but|instead|wrong|pr|issue|repo|#\d+)\b",
                         text, re.I)


def _off_world_coding(text: str, prior: str, agent_text: str) -> bool:
    """Git identifiers do not belong on a shopping or orders thread."""
    blob = f"{prior} {agent_text}"
    return bool(_CODING_MARK.search(text) and not _CODING_MARK.search(blob))


def _accept_followup(text: str, prior: str, agent_text: str) -> bool:
    from .generator import usable_user_message
    if not text or text == prior or len(text) > 2000:
        return False
    # A person never types call syntax: name_with_underscores( or a JSON dump.
    if re.search(r"\b\w+_\w+\s*\(", text) or text.count('"') >= 4:
        return False
    if _RUBBER_STAMP.match(text) or _CONFIRM_ONLY.match(text):
        return False
    if _mostly_thanks(text):
        return False
    if _off_world_coding(text, prior, agent_text):
        return False
    if _ID_FOLLOW.search(text) and len(text) >= 3:
        return True
    if _echoes_agent(text, agent_text):
        return False
    if len(text) < 8:
        return False
    return usable_user_message(text)


def _tool_world(tools: list | None) -> str:
    names: list[str] = []
    for item in tools or []:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = str((fn or {}).get("name") or "").strip()
        if name and name not in names:
            names.append(name)
    return ", ".join(names[:12])


def _user_followup(base_url: str, model: str, prior: str, agent_text: str, *,
                   api_key: str | None, timeout: float,
                   messages: list[dict] | None = None,
                   steps: list[dict] | None = None,
                   want: str = "",
                   tools: list | None = None) -> str:
    from .generator import (clean_user_message,
                            _realize_typed_message, _strip_directive_phrases)
    trace = _render_user_trace(messages, steps)
    if not trace:
        trace = (f"You said: {prior[:500]}\n"
                 f"The agent replied: {(agent_text or '')[:500]}")
    opening = str(want or prior).strip()
    persona = _persona_notes(opening)
    need = opening[:180] if opening else "this handled"
    who = f"You are the human who needs {need}."
    if persona:
        who = f"{who} {persona}"
    world = _tool_world(tools)
    if world:
        who = f"{who} This agent's tools are: {world}."
    asked = _agent_asked(agent_text)
    nudge = ""
    if asked:
        nudge = (
            " They asked you a question. Answer it with one concrete detail "
            "from this same world. One short line."
        )
    body = (
        f"This is what's been said so far:\n{trace}\n\n"
        f"Your turn to respond. {who}{nudge}"
    )
    retry_body = (
        f"This is what's been said so far:\n{trace}\n\n"
        "Answer their last question with one concrete detail from this "
        "same world. One short line. No thanks. No git repo unless this "
        "thread is already about git."
    )
    tags: dict = {}
    if persona and "every letter small" in persona:
        tags["texture"] = "lowercase"
    if persona and "Leave out end marks" in persona:
        tags["texture"] = "no_punctuation"
    if persona and "Capitalize normally" in persona and "texture" not in tags:
        tags["texture"] = "standard"
    wait = min(8.0 if asked else 5.0, float(timeout or 5.0))
    for attempt, content in enumerate((body, retry_body) if asked else (body,)):
        try:
            reply = complete(
                base_url, model,
                [{"role": "system", "content": _USER_SIM_SYSTEM},
                 {"role": "user", "content": content}],
                tools=None, api_key=api_key,
                temperature=(_RESPONSE_TEMP_LO + _RESPONSE_TEMP_HI) / 2,
                max_tokens=120 if attempt else 180, timeout=wait)
        except Exception:
            continue
        text = _scrub_ai_traces(_realize_typed_message(
            _strip_directive_phrases(clean_user_message(reply.get("content") or "")),
            tags))
        if _accept_followup(text, prior, agent_text):
            return text
    return ""


def _echoes_agent(user: str, agent: str) -> bool:
    """True when the user turn is mostly a restatement of the assistant.

    Shared identifiers (order ids, ticket numbers) are normal ping-pong,
    not an echo. Compare letter tokens only, and require a high overlap.
    """
    u = set(re.findall(r"[a-z]{3,}", (user or "").lower()))
    a = set(re.findall(r"[a-z]{3,}", (agent or "").lower()))
    if not u or not a:
        return False
    return (len(u & a) / len(u)) >= 0.70


def local_model(base_url: str, model: str, *, tools: list[dict],
                system: str = "", api_key: str | None = None,
                max_turns: int | None = None, avg_turns: float = 6,
                turn_stats: dict | None = None,
                temperature: float = 0.8,
                fault_plans: dict | None = None,
                result_shapes: dict | None = None,
                timeout: float = 60) -> Callable:
    local = threading.local()
    plans = fault_plans if fault_plans is not None else {}
    shapes = result_shapes if result_shapes is not None else {}
    cap = (default_max_turns(n_tools=len(tools))
           if max_turns is None else max(1, int(max_turns)))
    policy_text = str(system or "").strip()

    def agent(message: str) -> dict:
        # New chat every call. Prior turns, tools, and follow-ups do not carry over.
        for attr in ("messages", "steps", "history"):
            if hasattr(local, attr):
                delattr(local, attr)
        plan = dict(plans.get(message) or {})
        world = str(plan.pop("world_state", "") or "")
        local.env = MockEnvironment(tools, faults=plan, world_state=world,
                                    result_shapes=shapes)
        turns = [part.strip() for part in str(message).split("\n<USER_TURN>\n")
                 if part.strip()] or [str(message)]
        messages = ([{"role": "system", "content": policy_text}] if policy_text else []) + [
            {"role": "user", "content": turns[0]},
        ]
        steps: list[dict] = []
        user_turn = 0
        n_user = 1
        last_user = turns[0]
        final_text = ""
        budget = sample_turn_budget(
            0, message, cap, avg_turns=avg_turns,
            running_mean=running_turn_mean(turn_stats))
        remaining = budget
        turn_i = 0
        closing_bonus = False
        while remaining > 0:
            remaining -= 1
            turn_i += 1
            reply = complete(base_url, model, messages, tools=tools, api_key=api_key,
                             temperature=temperature, timeout=timeout,
                             max_tokens=768)
            calls, assistant = _calls_from_reply(reply)
            spoken = (_spoken_text(reply) or "").strip()
            if spoken:
                final_text = spoken
            if not spoken and not calls:
                if remaining > 0:
                    continue
                return _finish_on_agent(steps, final_text)
            if calls:
                messages.append(assistant)
                attached = False
                for call in calls:
                    fn = call.get("function", {})
                    try:
                        arguments = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    result = local.env.call(fn.get("name", ""), arguments)
                    step = {"tool": fn.get("name", ""), "arguments": arguments,
                            "result": result}
                    if spoken and not attached:
                        step["text"] = spoken
                        attached = True
                    steps.append(step)
                    messages.append({"role": "tool",
                                     "tool_call_id": call.get("id", ""),
                                     "content": json.dumps(result)})
                if remaining <= 0 and not closing_bonus:
                    remaining = 1
                    closing_bonus = True
                continue
            if spoken:
                prev = ""
                for s in reversed(steps):
                    if isinstance(s, dict) and str(s.get("text") or "").strip():
                        prev = str(s["text"]).strip()
                        break
                if prev and spoken.strip() == prev and n_user >= 2:
                    return _finish_on_agent(steps, prev)
            messages.append({"role": "assistant", "content": spoken})
            if spoken:
                steps.append({"text": spoken})
            # After the agent speaks, the user replies only if the thread
            # is still open. Do not stack assistant-only variants of the
            # same line. One extra agent beat is allowed only for a short
            # preamble before the first tool call.
            if (not _has_tool_steps(steps)
                    and _looks_unfinished(spoken, steps)
                    and remaining > 0):
                continue
            room = remaining > 0
            if room and user_turn + 1 < len(turns):
                user_turn += 1
                n_user += 1
                last_user = turns[user_turn]
                steps.append({"user": last_user})
                messages.append({"role": "user", "content": last_user})
                continue
            need_first = n_user < 2
            if (room or need_first) and _want_followup(
                    message, turn_i, user_turns=n_user, budget=budget,
                    agent_text=spoken):
                follow = _user_followup(
                    base_url, model, last_user, spoken,
                    api_key=api_key, timeout=timeout,
                    messages=messages, steps=steps, want=turns[0],
                    tools=tools)
                if follow:
                    last_user = follow
                    n_user += 1
                    steps.append({"user": follow})
                    messages.append({"role": "user", "content": follow})
                    if remaining <= 0:
                        remaining = 1
                    continue
            return _finish_on_agent(steps, spoken or final_text)
        return _finish_on_agent(steps, final_text)

    agent.__name__ = f"local_model[{model}]"
    agent.fault_plans = plans
    agent.system = policy_text
    agent.policy = policy_text
    return agent


def hosted_model(tools: list[dict], system: str = "",
                 fault_plans: dict | None = None, **kwargs) -> Callable:
    """The default simulation brain: hosted Qwen wearing these tools."""
    url, model = parse_backend_spec(default_agent_spec())
    return local_model(url, model, tools=tools, system=system,
                       fault_plans=fault_plans, **kwargs)
