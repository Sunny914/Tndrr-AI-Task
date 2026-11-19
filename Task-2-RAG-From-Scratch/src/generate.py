from __future__ import annotations

import requests

from src.config import (
    OLLAMA_GENERATE_URL,
    OLLAMA_MODEL,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TAGS_URL,
    OLLAMA_TIMEOUT_SECONDS,
    TOP_K,
)
from src.retrieve import retrieve


PROMPT_INSTRUCTIONS = """You are answering questions about the NIST Artificial Intelligence Risk Management Framework (AI RMF 1.0).

Use ONLY the evidence excerpts below. Do not use outside knowledge.
If the evidence is insufficient, say that the provided evidence is insufficient.
Do not invent facts, numbers, definitions, or citations that are not in the evidence.
Cite physical PDF page numbers from the evidence labels when you use them.
"""


def ollama_available(timeout: float = 5.0) -> tuple[bool, str]:
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=timeout)
        response.raise_for_status()
        names = [m.get("name", "") for m in response.json().get("models", [])]
        if any(name.startswith(OLLAMA_MODEL) for name in names):
            return True, f"Ollama reachable; model {OLLAMA_MODEL} is registered."
        return False, (
            f"Ollama reachable, but model {OLLAMA_MODEL!r} was not found. "
            f"Registered models: {names}"
        )
    except requests.RequestException as exc:
        return False, f"Ollama unavailable: {exc}"


def build_prompt(question: str, retrieved: list[dict]) -> str:
    evidence_blocks = []
    for item in retrieved:
        evidence_blocks.append(
            f"[PDF page {item['page_number']} | {item['chunk_id']} | score={item['score']:.4f}]\n"
            f"{item['text']}"
        )
    evidence = "\n\n".join(evidence_blocks) if evidence_blocks else "(no evidence retrieved)"
    return (
        f"{PROMPT_INSTRUCTIONS}\n"
        f"Evidence:\n{evidence}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def generate_answer(question: str, retrieved: list[dict]) -> dict:
    available, status = ollama_available()
    if not available:
        return {
            "answer": None,
            "model": OLLAMA_MODEL,
            "ollama_available": False,
            "error": status,
        }

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": build_prompt(question, retrieved),
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": OLLAMA_NUM_PREDICT,
        },
    }
    try:
        response = requests.post(
            OLLAMA_GENERATE_URL,
            json=payload,
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        answer = (data.get("response") or "").strip()
        if not answer:
            return {
                "answer": None,
                "model": OLLAMA_MODEL,
                "ollama_available": True,
                "error": "Ollama returned an empty response.",
            }
        return {
            "answer": answer,
            "model": OLLAMA_MODEL,
            "ollama_available": True,
            "error": None,
        }
    except requests.RequestException as exc:
        return {
            "answer": None,
            "model": OLLAMA_MODEL,
            "ollama_available": False,
            "error": f"Ollama generation failed: {exc}",
        }


def answer_question(question: str, config_id: str = "A", top_k: int = TOP_K) -> dict:
    retrieved = retrieve(question, config_id=config_id, top_k=top_k)
    generation = generate_answer(question, retrieved)
    retrieved_pages = [item["page_number"] for item in retrieved]
    return {
        "question": question,
        "config_id": config_id,
        "answer": generation["answer"],
        "ollama_available": generation["ollama_available"],
        "error": generation["error"],
        "model": generation["model"],
        "retrieved": retrieved,
        "retrieved_pages": retrieved_pages,
        "retrieved_chunk_ids": [item["chunk_id"] for item in retrieved],
        "note": (
            "retrieved_pages come from the retriever, not from the generated answer. "
            "Do not treat LLM citations as verified sources."
        ),
    }
