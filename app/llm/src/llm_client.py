import httpx
import json


class OllamaClient:
    def __init__(self, base_url="http://localhost:11434", model="llama3.1:8b",
                 num_ctx=16384, temperature_report=0, temperature_chat=0.3,
                 num_gpu=-1):
        self.base_url = base_url
        self.model = model
        self.num_ctx = num_ctx
        self.temperature_report = temperature_report  # 0 = deterministic, for one-shot report
        self.temperature_chat   = temperature_chat    # 0.3 = natural conversation
        self.num_gpu = num_gpu                        # -1 = all layers on GPU
        self.timeout = httpx.Timeout(300.0, connect=10.0)

    def is_available(self):
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return r.status_code == 200
        except httpx.ConnectError:
            return False

    def list_models(self):
        r = httpx.get(f"{self.base_url}/api/tags", timeout=10.0)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def _base_options(self, temperature: float) -> dict:
        opts = {
            "temperature": temperature,
            "num_ctx": self.num_ctx,
        }
        if self.num_gpu != 0:
            opts["num_gpu"] = self.num_gpu
        return opts

    # ── generate (one-shot, report mode) ──────────────────────────────────────

    def generate(self, prompt, system=None, stream_callback=None):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream_callback is not None,
            "options": self._base_options(self.temperature_report),
        }
        if system:
            payload["system"] = system
        return (self._generate_stream(payload, stream_callback)
                if stream_callback else self._generate_sync(payload))

    def _generate_sync(self, payload):
        payload["stream"] = False
        r = httpx.post(f"{self.base_url}/api/generate",
                       json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["response"]

    def _generate_stream(self, payload, callback):
        payload["stream"] = True
        full = []
        with httpx.stream("POST", f"{self.base_url}/api/generate",
                          json=payload, timeout=self.timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    full.append(token)
                    callback(token)
                if chunk.get("done", False):
                    break
        return "".join(full)

    # ── chat (conversational mode) ─────────────────────────────────────────────

    def chat(self, messages, stream_callback=None):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream_callback is not None,
            "options": self._base_options(self.temperature_chat),
        }
        return (self._chat_stream(payload, stream_callback)
                if stream_callback else self._chat_sync(payload))

    def _chat_sync(self, payload):
        payload["stream"] = False
        r = httpx.post(f"{self.base_url}/api/chat",
                       json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def _chat_stream(self, payload, callback):
        payload["stream"] = True
        full = []
        with httpx.stream("POST", f"{self.base_url}/api/chat",
                          json=payload, timeout=self.timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    full.append(token)
                    callback(token)
                if chunk.get("done", False):
                    break
        return "".join(full)
