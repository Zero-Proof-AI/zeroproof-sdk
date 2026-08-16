"""Scenario-embedding selection. Lexical and semantic vectors are never mixed."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Iterable, Sequence

from .diversity import apply_annealing_explore
from .scenarios import novelty as min_cosine_distance

_DIM = 256


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _hash_vector(text: str, dim: int = _DIM) -> list[float]:
    vec = [0.0] * dim
    normalized = " ".join(str(text).lower().split())
    words = normalized.split()
    tokens = [f"w:{w}" for w in words] + [
        f"b:{a}>{b}" for a, b in zip(words, words[1:])]
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        vec[int.from_bytes(digest, "big") % dim] += 1.0
    return _normalize(vec)


class HashEmbedder:
    name = "hash-word-bigram"
    semantic = False

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [_hash_vector(text) for text in texts]


class CallableEmbedder:
    def __init__(self, fn: Callable[[Sequence[str]], Any], name: str = "callable",
                 semantic: bool = True):
        self.fn = fn
        self.name = name
        self.semantic = semantic

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raw = self.fn(list(texts))
        return [_normalize([float(x) for x in row]) for row in raw]


class OllamaEmbedder:
    semantic = True

    def __init__(self, model: str = "nomic-embed-text",
                 host: str = "http://127.0.0.1:11434"):
        self.model = model
        self.host = host.rstrip("/")
        self.name = f"ollama:{model}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        output: list[list[float]] = []
        for start in range(0, len(texts), 128):
            request = urllib.request.Request(
                f"{self.host}/api/embed",
                data=json.dumps({"model": self.model,
                                 "input": list(texts[start:start + 128])}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                output.extend(json.loads(response.read()).get("embeddings") or [])
        if len(output) != len(texts):
            raise RuntimeError(
                f"{self.name} returned {len(output)} embeddings for {len(texts)} texts")
        return [_normalize([float(x) for x in row]) for row in output]


class OpenAIEmbedder:
    semantic = True

    def __init__(self, model: str = "text-embedding-3-small",
                 base_url: str = "https://api.openai.com/v1", api_key: str | None = None):
        self.model = model
        self.base_url = os.environ.get("OPENAI_BASE_URL", base_url).rstrip("/")
        self.api_key = (api_key or os.environ.get("OPENAI_API_KEY")
                        or os.environ.get("ZEROPROOF_API_KEY", ""))
        self.name = f"openai:{model}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        output: list[list[float]] = []
        for start in range(0, len(texts), 128):
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            request = urllib.request.Request(
                f"{self.base_url}/embeddings",
                data=json.dumps({"model": self.model,
                                 "input": list(texts[start:start + 128])}).encode(),
                headers=headers,
            )
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(request, timeout=30) as response:
                        rows = sorted(json.loads(response.read()).get("data") or [],
                                      key=lambda row: row["index"])
                        output.extend(row["embedding"] for row in rows)
                    break
                except (OSError, urllib.error.URLError):
                    if attempt == 2:
                        raise
                    time.sleep(1.5 * (attempt + 1))
        if len(output) != len(texts):
            raise RuntimeError(
                f"{self.name} returned {len(output)} embeddings for {len(texts)} texts")
        return [_normalize([float(x) for x in row]) for row in output]


DEFAULT_EMBED_URL = (
    "https://zeroproofai--zeroproof-embed-embedder-embed.modal.run")


class ModalEmbedder:
    """The hosted Zero Proof embedding endpoint: batched, semantic."""

    def __init__(self, url: str | None = None, api_key: str | None = None,
                 timeout: float = 20):
        self.url = (url or os.environ.get("ZEROPROOF_EMBED")
                    or DEFAULT_EMBED_URL)
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "")
        self.timeout = timeout
        self.name = "modal:bge-small"
        self.semantic = True

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        body = json.dumps({"texts": list(map(str, texts))}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())["embeddings"]


def _live_modal(url: str | None = None) -> ModalEmbedder | None:
    candidate = ModalEmbedder(url, timeout=1.5)
    try:
        candidate.embed(["probe"])
    except Exception:
        return None
    candidate.timeout = 20
    return candidate


def resolve_embedder(embedder: Any = "hash") -> Any:
    if embedder is None or embedder == "hash":
        return HashEmbedder()
    if embedder == "modal":
        return _live_modal() or HashEmbedder()
    if isinstance(embedder, str):
        kind, _, rest = embedder.partition(":")
        if kind == "ollama":
            return OllamaEmbedder(rest or "nomic-embed-text")
        if kind == "openai":
            return OpenAIEmbedder(rest or "text-embedding-3-small")
        if kind == "modal":
            return _live_modal(rest or None) or HashEmbedder()
        raise ValueError("embedder must be hash, ollama:<model>, openai:<model>, or a callable")
    if hasattr(embedder, "embed"):
        return embedder
    if callable(embedder):
        return CallableEmbedder(embedder)
    raise TypeError("invalid embedder")


def is_semantic(embedder: Any) -> bool:
    return bool(getattr(embedder, "semantic", False)) and not str(
        getattr(embedder, "name", "")).startswith("hash")


def kmeans(vectors: list[list[float]], k: int, seed: int,
           iterations: int = 40) -> tuple[list[int], list[list[float]]]:
    n = len(vectors)
    if not n:
        return [], []
    dim = len(vectors[0])
    k = max(1, min(k, n))
    rng = random.Random(seed)
    centers = [list(vectors[rng.randrange(n)])]
    while len(centers) < k:
        distances = []
        for vec in vectors:
            distances.append(min(sum((a - b) ** 2 for a, b in zip(vec, c))
                                 for c in centers))
        total = sum(distances)
        if total <= 0:
            centers.append(list(vectors[len(centers) % n]))
            continue
        pick = rng.random() * total
        acc = 0.0
        index = n - 1
        for i, dist in enumerate(distances):
            acc += dist
            if acc >= pick:
                index = i
                break
        centers.append(list(vectors[index]))
    labels = [-1] * n
    for _ in range(iterations):
        new_labels = []
        for vec in vectors:
            best, best_d = 0, None
            for ci, center in enumerate(centers):
                d = sum((a - b) ** 2 for a, b in zip(vec, center))
                if best_d is None or d < best_d:
                    best, best_d = ci, d
            new_labels.append(best)
        if new_labels == labels:
            break
        labels = new_labels
        for cluster in range(k):
            members = [vectors[i] for i, lab in enumerate(labels) if lab == cluster]
            if members:
                centers[cluster] = [sum(row[j] for row in members) / len(members)
                                    for j in range(dim)]
    return labels, centers


def select_diverse(vectors: list[list[float]], k: int) -> list[int]:
    if k >= len(vectors):
        return list(range(len(vectors)))
    if not vectors:
        return []
    dim = len(vectors[0])
    centroid = [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]
    centroid = _normalize(centroid)
    first = min(range(len(vectors)), key=lambda i: _cos(vectors[i], centroid))
    chosen = [first]
    nearest = [1.0 - _cos(v, vectors[first]) for v in vectors]
    chosen_set = {first}
    while len(chosen) < k:
        nxt = max((i for i in range(len(vectors)) if i not in chosen_set),
                  key=lambda i: nearest[i], default=None)
        if nxt is None:
            break
        chosen.append(nxt)
        chosen_set.add(nxt)
        for i, v in enumerate(vectors):
            d = 1.0 - _cos(v, vectors[nxt])
            if d < nearest[i]:
                nearest[i] = d
    return chosen


class EmbeddingArchive:
    """Tested scenario embeddings. Refuses to mix lexical and semantic rows."""

    def __init__(self, embedder_name: str, semantic: bool):
        self.embedder_name = embedder_name
        self.semantic = semantic
        self.vectors: list[list[float]] = []

    def compatible(self, embedder: Any) -> bool:
        return (str(getattr(embedder, "name", "")) == self.embedder_name
                and bool(is_semantic(embedder)) == self.semantic)

    def add(self, vectors: Iterable[list[float]]) -> None:
        self.vectors.extend(vectors)


def select_execution_batch(candidates: list[str], *, embedder: Any,
                           archive: EmbeddingArchive, batch_size: int,
                           seed: int = 0, round_index: int = 0) -> tuple[list[dict], dict]:
    """Embed, cluster/stratify, k-center, novelty-filter, return batch metas.

    Each meta: text, vector, cluster, novelty, reason.
    """
    texts = list(dict.fromkeys(str(c) for c in candidates if str(c)))
    info = {
        "embedder": getattr(embedder, "name", "unknown"),
        "semantic": is_semantic(embedder),
        "degraded": [],
        "mixed_spaces_refused": False,
    }
    if not texts:
        return [], info
    vectors = embedder.embed(texts)
    if len(vectors) != len(texts):
        raise RuntimeError("embedder returned a different count than inputs")

    n_clusters = max(1, min(8, round(math.sqrt(len(texts)))))
    labels, _ = kmeans(vectors, n_clusters, seed, iterations=8)
    by_cluster: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        by_cluster.setdefault(int(label), []).append(i)

    need = max(1, int(batch_size))
    fill_n = max(1, need // 2)
    typical_n = max(0, need - fill_n)

    archive_ok = archive.compatible(embedder)
    if archive.vectors and not archive_ok:
        info["mixed_spaces_refused"] = True
        info["degraded"].append("embedding_space_mismatch")
        tested = []
    else:
        tested = archive.vectors

    rng = random.Random(seed)
    typical_idx: list[int] = []
    cluster_order = sorted(by_cluster, key=lambda c: (-len(by_cluster[c]), c))
    while len(typical_idx) < typical_n and cluster_order:
        progressed = False
        for cluster in cluster_order:
            remaining = [i for i in by_cluster[cluster] if i not in typical_idx]
            if not remaining:
                continue
            typical_idx.append(rng.choice(remaining))
            progressed = True
            if len(typical_idx) >= typical_n:
                break
        if not progressed:
            break

    per_cluster = max(1, math.ceil((need * 2) / max(1, len(by_cluster))))
    stratified: list[int] = []
    for indices in by_cluster.values():
        sub = [vectors[i] for i in indices]
        picks = select_diverse(sub, min(per_cluster, len(indices)))
        stratified.extend(indices[p] for p in picks)
    stratified = list(dict.fromkeys(stratified))
    spread_vecs = [vectors[i] for i in stratified]
    spread_picks = select_diverse(spread_vecs, min(max(need * 2, need),
                                                   len(stratified)))
    ranked = [stratified[p] for p in spread_picks]

    taken = set(typical_idx)
    fill_idx: list[int] = []
    novelty_of = {}
    for index in ranked:
        novelty_of[index] = min_cosine_distance(vectors[index], tested)
    for index in sorted(ranked, key=lambda i: (-novelty_of.get(i, 0.0), texts[i])):
        if index in taken:
            continue
        fill_idx.append(index)
        taken.add(index)
        if len(fill_idx) >= fill_n:
            break
    for index in range(len(texts)):
        if len(typical_idx) + len(fill_idx) >= need:
            break
        if index not in taken:
            fill_idx.append(index)
            taken.add(index)

    batch_idx = list(dict.fromkeys(typical_idx + fill_idx))[:need]
    batch_idx = apply_annealing_explore(
        batch_idx, len(texts), novelty_of, need=need,
        round_index=round_index, seed=seed)
    selected: list[dict] = []
    for index in batch_idx:
        score = novelty_of.get(index)
        if score is None:
            score = min_cosine_distance(vectors[index], tested)
        kind = "typical" if index in typical_idx else "fill"
        selected.append({
            "text": texts[index],
            "vector": vectors[index],
            "cluster": int(labels[index]),
            "novelty": score,
            "reason": (f"{kind} cluster={labels[index]} novelty={score:.3f} "
                       f"embedder={info['embedder']}"),
        })
    info["clusters"] = len(by_cluster)
    info["candidate_pool"] = len(texts)
    info["batch_design"] = "typical_kcenter_fill_plus_annealing_explore"
    info["round_index"] = int(round_index)
    dense = cluster_order[: max(1, len(cluster_order) // 3)] if cluster_order else []
    sparse = list(reversed(cluster_order))[: max(1, len(cluster_order) // 3)] if cluster_order else []
    info["concentrated"] = [texts[by_cluster[c][0]] for c in dense if by_cluster[c]][:8]
    info["sparse"] = [texts[by_cluster[c][0]] for c in sparse if by_cluster[c]][:8]
    return selected, info
