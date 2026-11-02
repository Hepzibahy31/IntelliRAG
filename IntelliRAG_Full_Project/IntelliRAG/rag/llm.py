"""LLM access: Groq (free), OpenAI, or local Ollama. Without a key the app still works
in retrieval-only mode."""
import requests
from . import config


class LLMError(Exception):
    pass


def provider() -> str:
    choice = config.get("LLM_PROVIDER", "auto").lower()
    if choice in ("groq", "openai", "ollama", "none"):
        return choice
    if config.get("GROQ_API_KEY"):
        return "groq"
    if config.get("OPENAI_API_KEY"):
        return "openai"
    return "none"


def model_name() -> str:
    p = provider()
    if p == "groq":
        return config.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    if p == "openai":
        return config.get("OPENAI_MODEL", "gpt-4o-mini")
    if p == "ollama":
        return config.get("OLLAMA_MODEL", "llama3.2")
    return ""


def available() -> bool:
    p = provider()
    if p == "groq":
        return bool(config.get("GROQ_API_KEY"))
    if p == "openai":
        return bool(config.get("OPENAI_API_KEY"))
    return p == "ollama"


def info() -> dict:
    return {"provider": provider(), "model": model_name(), "available": available()}


def chat(messages, temperature=0.2, max_tokens=800) -> str:
    p = provider()
    try:
        if p == "ollama":
            url = config.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
            r = requests.post(f"{url}/api/chat", timeout=180, json={
                "model": model_name(), "messages": messages, "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens}})
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        if p in ("groq", "openai"):
            key = config.get("GROQ_API_KEY" if p == "groq" else "OPENAI_API_KEY")
            if not key:
                raise LLMError(f"{p.upper()} API key is missing.")
            url = ("https://api.groq.com/openai/v1/chat/completions" if p == "groq"
                   else "https://api.openai.com/v1/chat/completions")
            body = {"model": model_name(), "messages": messages}
            if p == "groq":
                body.update(temperature=temperature, max_tokens=max_tokens)
            else:
                body["max_completion_tokens"] = max_tokens
            r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=60)
            if r.status_code != 200:
                raise LLMError(f"{p} API error {r.status_code}: {r.text[:200]}")
            return r.json()["choices"][0]["message"]["content"].strip()
    except requests.RequestException as error:
        raise LLMError(f"Could not reach the {p} service: {error}")
    raise LLMError("No LLM configured.")
