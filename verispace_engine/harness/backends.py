"""Model backends behind one interface: generate(messages) -> text.

HFBackend      local Hugging Face transformers (CPU or GPU); imported lazily.
OpenAICompatibleBackend  any /chat/completions endpoint; the key comes from the environment or a
                         .env file and is never logged or written anywhere.
ScriptedBackend  replays fixed replies; used in tests.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Protocol, Sequence

Message = dict[str, str]                       # {"role": "system"|"user"|"assistant", "content": ...}


class ModelBackend(Protocol):
    name: str

    def generate(self, messages: Sequence[Message], max_new_tokens: int = 256, temperature: float = 0.0,
                 seed: int | None = None) -> str: ...


def load_dotenv_value(key: str, path: str | Path = ".env") -> str | None:
    """Read one value from the environment, else from a .env file. Never prints it."""
    if os.environ.get(key):
        return os.environ[key]
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            k, sep, v = line.partition("=")
            if sep and k.strip() == key:
                return v.strip().strip("\"'") or None
    return None


class ScriptedBackend:
    """Replies from a list (cycled per call) or a function of the message list."""

    def __init__(self, replies: Sequence[str] | Callable[[Sequence[Message]], str], name: str = "scripted") -> None:
        self.name, self._replies, self._i = name, replies, 0

    def generate(self, messages, max_new_tokens=256, temperature=0.0, seed=None) -> str:
        if callable(self._replies):
            return self._replies(messages)
        reply = self._replies[min(self._i, len(self._replies) - 1)]
        self._i += 1
        return reply


class HFBackend:
    def __init__(self, model_id: str, device: str | None = None, dtype: str | None = None,
                 adapter: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:             # pragma: no cover - environment dependent
            raise RuntimeError("local models need: pip install transformers (see the 'harness' extra)") from exc
        self.name = model_id
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype or (torch.float32 if self.device == "cpu" else torch.float16)).to(self.device)
        if adapter:                                  # a LoRA adapter folder from verispace_engine.sft
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
            self.name = f"{model_id}+{Path(adapter).parent.name}"
        self.model = model.eval()

    def generate(self, messages, max_new_tokens=256, temperature=0.0, seed=None) -> str:
        torch = self._torch
        ids = self.tokenizer.apply_chat_template(list(messages), add_generation_prompt=True, return_tensors="pt",
                                                 return_dict=True).to(self.device)
        if seed is not None:
            torch.manual_seed(seed)
        kwargs = {"do_sample": True, "temperature": temperature, "top_p": 0.95} if temperature > 0 else {"do_sample": False}
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id,
                                      **kwargs)
        return self.tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


class OpenAICompatibleBackend:
    """POST {base_url}/chat/completions. Key variable defaults to OPENAI_API_KEY (env or .env)."""

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1", key_var: str = "OPENAI_API_KEY",
                 timeout: float = 120.0, post=None) -> None:
        self.name, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout
        self._key = load_dotenv_value(key_var)
        if post is None:
            import requests
            post = requests.post
        self._post = post

    def generate(self, messages, max_new_tokens=256, temperature=0.0, seed=None) -> str:
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        body = {"model": self.name, "messages": list(messages), "max_tokens": max_new_tokens,
                "temperature": temperature}
        if seed is not None:
            body["seed"] = seed
        resp = self._post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"endpoint returned HTTP {resp.status_code}")   # body omitted: may echo request data
        return resp.json()["choices"][0]["message"]["content"] or ""
