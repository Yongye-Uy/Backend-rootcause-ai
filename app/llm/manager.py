import logging
from dataclasses import dataclass

from app.config import settings as default_settings
from app.llm.providers import BaseProvider, ProviderError, build_providers

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    content: str
    provider: str


class AllProvidersFailedError(Exception):
    def __init__(self, attempts: dict[str, str]):
        self.attempts = attempts
        summary = "; ".join(f"{name}: {reason}" for name, reason in attempts.items())
        super().__init__(f"All LLM providers failed: {summary}")


class LLMManager:
    def __init__(
        self,
        providers: dict[str, BaseProvider] | None = None,
        order: list[str] | None = None,
    ):
        self.providers = providers if providers is not None else build_providers(default_settings)
        self.order = order if order is not None else default_settings.provider_order
        self.last_provider: str | None = None

    def generate(self, messages: list[dict], stop: list[str] | None = None) -> GenerationResult:
        attempts: dict[str, str] = {}
        for name in self.order:
            provider = self.providers.get(name)
            if provider is None:
                attempts[name] = "not configured"
                continue
            try:
                content = provider.generate(messages, stop=stop)
                self.last_provider = name
                return GenerationResult(content=content, provider=name)
            except ProviderError as e:
                logger.warning("LLM provider %s failed: %s", name, e)
                attempts[name] = str(e)
                continue
        raise AllProvidersFailedError(attempts)


_default_manager: LLMManager | None = None


def get_llm_manager() -> LLMManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = LLMManager()
    return _default_manager
