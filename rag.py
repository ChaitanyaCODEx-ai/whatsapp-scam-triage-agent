"""RAG layer: ChromaDB (in-memory) + local sentence-transformers embeddings.

No embedding API key needed. Two collections:
  - "scams": known scam / legitimate examples (data/scam_examples.json)
  - "kb":    policies and privacy-settings tips (data/policies_and_tips.txt)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Some hosts ship an old sqlite3 that Chroma rejects; use pysqlite3 if present.
try:
    import pysqlite3  # type: ignore  # noqa: F401

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb
from chromadb.config import Settings

DATA_DIR = Path(__file__).parent / "data"
EMBED_MODEL = "all-MiniLM-L6-v2"
_state: dict = {}


def _embedding_fn():
    """Prefer sentence-transformers; fall back to Chroma's ONNX build of the same model."""
    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        return SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL), "sentence-transformers"
    except Exception:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        return DefaultEmbeddingFunction(), "chroma-default-onnx (all-MiniLM-L6-v2)"


def load_scams() -> list[dict]:
    return json.loads((DATA_DIR / "scam_examples.json").read_text(encoding="utf-8"))


def load_kb() -> list[dict]:
    """Parse '### ID | KIND | tags: a, b | Title' blocks from the text file."""
    items: list[dict] = []
    cur = None
    for line in (DATA_DIR / "policies_and_tips.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("### "):
            if cur:
                items.append(cur)
            id_, kind, tags, title = [p.strip() for p in line[4:].split("|", 3)]
            cur = {
                "id": id_,
                "kind": kind,
                "tags": [t.strip() for t in tags.replace("tags:", "").split(",") if t.strip()],
                "title": title,
                "body": [],
            }
        elif cur is not None and line.strip() and not line.startswith("#"):
            cur["body"].append(line.strip())
    if cur:
        items.append(cur)
    for it in items:
        it["body"] = " ".join(it["body"])
    return items


def get_store():
    """Build (once per process) and return (scams_collection, kb_collection)."""
    if "scams" in _state:
        return _state["scams"], _state["kb"]

    ef, backend = _embedding_fn()
    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    cos = {"hnsw:space": "cosine"}
    scams = client.get_or_create_collection("scams", embedding_function=ef, metadata=cos)
    kb = client.get_or_create_collection("knowledge_base", embedding_function=ef, metadata=cos)

    if scams.count() == 0:
        rows = load_scams()
        scams.add(
            ids=[r["id"] for r in rows],
            documents=[r["message"] for r in rows],
            metadatas=[
                {
                    "category": r["category"],
                    "severity": r["severity"],
                    "action": r["action"],
                    "red_flags": " | ".join(r["red_flags"]),
                }
                for r in rows
            ],
        )
    if kb.count() == 0:
        items = load_kb()
        kb.add(
            ids=[i["id"] for i in items],
            documents=[f"{i['title']}. {i['body']}" for i in items],
            metadatas=[
                {"kind": i["kind"], "tags": ",".join(i["tags"]), "title": i["title"], "body": i["body"]}
                for i in items
            ],
        )
    _state.update(scams=scams, kb=kb, backend=backend)
    return scams, kb


def backend_name() -> str:
    get_store()
    return _state["backend"]


def list_categories() -> list[str]:
    return sorted({r["category"] for r in load_scams() if r["category"] != "legitimate"})


def retrieve_similar_scams(text: str, k: int = 3) -> list[dict]:
    """Top-k most similar known examples (scam or legitimate) with cosine similarity."""
    scams, _ = get_store()
    res = scams.query(query_texts=[text], n_results=min(k, scams.count()))
    out = []
    for i, id_ in enumerate(res["ids"][0]):
        md = res["metadatas"][0][i]
        out.append(
            {
                "id": id_,
                "message": res["documents"][0][i],
                "category": md["category"],
                "severity": md["severity"],
                "action": md["action"],
                "red_flags": [f for f in md["red_flags"].split(" | ") if f],
                "similarity": round(max(0.0, 1 - res["distances"][0][i]), 3),
            }
        )
    return out


def retrieve_tips(text: str, scam_type: str | None = None, k: int = 6) -> list[dict]:
    """Semantic search over privacy tips, re-ranked toward tips tagged for this scam type."""
    _, kb = get_store()
    res = kb.query(query_texts=[text], n_results=min(12, kb.count()), where={"kind": "TIP"})
    ranked = []
    for i, id_ in enumerate(res["ids"][0]):
        md = res["metadatas"][0][i]
        tags = md["tags"].split(",")
        score = 1 - res["distances"][0][i]
        if scam_type and scam_type in tags:
            score += 0.25
        elif "all" in tags:
            score += 0.05
        ranked.append(
            {
                "id": id_,
                "title": md["title"],
                "body": md["body"],
                "tags": tags,
                "type": "setting" if id_.startswith("TS") else "habit",
                "score": round(score, 3),
            }
        )
    ranked.sort(key=lambda d: d["score"], reverse=True)
    return ranked[:k]


def get_policy(policy_id: str) -> dict:
    _, kb = get_store()
    res = kb.get(ids=[policy_id])
    if not res["ids"]:
        return {"id": policy_id, "title": "", "body": ""}
    md = res["metadatas"][0]
    return {"id": policy_id, "title": md["title"], "body": md["body"]}


if __name__ == "__main__":
    q = "Send me the 6 digit code you just received"
    print("backend:", backend_name())
    for r in retrieve_similar_scams(q):
        print(r["similarity"], r["category"], "-", r["message"][:70])
    for t in retrieve_tips(q, "otp_verification", 3):
        print(t["score"], t["id"], t["title"])
