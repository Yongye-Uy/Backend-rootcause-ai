import requests


class ProviderError(Exception):
    pass


class BaseProvider:
    name: str

    def generate(self, messages: list[dict], stop: list[str] | None = None) -> str:
        raise NotImplementedError


class OpenAICompatProvider(BaseProvider):
    def __init__(self, name: str, base_url: str, api_key: str, model: str, timeout: int = 120):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @property
    def _chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def generate(self, messages: list[dict], stop: list[str] | None = None) -> str:
        if not self.api_key:
            raise ProviderError(f"{self.name}: no API key configured")
        payload: dict = {"model": self.model, "messages": messages}
        if stop:
            payload["stop"] = stop
        try:
            response = requests.post(
                self._chat_completions_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise ProviderError(f"{self.name}: request failed ({e})") from e

        if response.status_code != 200:
            raise ProviderError(f"{self.name}: HTTP {response.status_code} {response.text[:300]}")

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ProviderError(f"{self.name}: unexpected response shape ({e})") from e

        if not content:
            raise ProviderError(f"{self.name}: empty response content")
        return content


def build_providers(settings) -> dict[str, BaseProvider]:
    return {
        "nvidia": OpenAICompatProvider(
            "nvidia", settings.nvidia_base_url, settings.nvidia_api_key, settings.nvidia_model
        ),
        "openrouter": OpenAICompatProvider(
            "openrouter",
            settings.openrouter_base_url,
            settings.openrouter_api_key,
            settings.openrouter_model,
        ),
        "ollama": OpenAICompatProvider(
            "ollama",
            settings.ollama_base_url.rstrip("/") + "/v1",
            settings.ollama_api_key,
            settings.ollama_model,
        ),
        "huggingface": OpenAICompatProvider(
            "huggingface",
            settings.huggingface_base_url,
            settings.huggingface_api_key,
            settings.huggingface_model,
        ),
        "gemini": OpenAICompatProvider(
            "gemini", settings.gemini_base_url, settings.gemini_api_key, settings.gemini_model
        ),
    }
