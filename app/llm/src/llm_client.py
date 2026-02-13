import httpx
import json


class OllamaClient:
    def __init__(self, base_url="http://localhost:11434", model="llama3.1:8b"):
        self.base_url = base_url
        self.model = model
        self.timeout = httpx.Timeout(300.0, connect=10.0)

    def is_available(self):
        """Check if Ollama server is running."""
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return r.status_code == 200
        except httpx.ConnectError:
            return False

    def list_models(self):
        """List available models."""
        r = httpx.get(f"{self.base_url}/api/tags", timeout=10.0)
        r.raise_for_status()
        data = r.json()
        return [m["name"] for m in data.get("models", [])]

    def generate(self, prompt, system=None, stream_callback=None):
        """
        Generate a response from the model.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            stream_callback: Optional callback(chunk_text) for streaming output
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream_callback is not None,
            "options": {
                "temperature": 0,
                "num_ctx": 8192
            }
        }
        if system:
            payload["system"] = system

        if stream_callback:
            return self._generate_stream(payload, stream_callback)
        else:
            return self._generate_sync(payload)

    def _generate_sync(self, payload):
        """Non-streaming generation."""
        payload["stream"] = False
        r = httpx.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()["response"]

    def _generate_stream(self, payload, callback):
        """Streaming generation with per-token callback."""
        payload["stream"] = True
        full_response = []

        with httpx.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    full_response.append(token)
                    callback(token)
                if chunk.get("done", False):
                    break

        return "".join(full_response)

    def chat(self, messages, stream_callback=None):
        """
        Chat-style conversation.

        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": "..."}
            stream_callback: Optional callback(chunk_text) for streaming
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream_callback is not None,
            "options": {
                "temperature": 0,
                "num_ctx": 8192
            }
        }

        if stream_callback:
            return self._chat_stream(payload, stream_callback)
        else:
            return self._chat_sync(payload)

    def _chat_sync(self, payload):
        payload["stream"] = False
        r = httpx.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    def _chat_stream(self, payload, callback):
        payload["stream"] = True
        full_response = []

        with httpx.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    full_response.append(token)
                    callback(token)
                if chunk.get("done", False):
                    break

        return "".join(full_response)
