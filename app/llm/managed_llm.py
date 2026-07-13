from typing import Any

from crewai import BaseLLM

from app.llm.manager import LLMManager


class ManagedLLM(BaseLLM):
    """CrewAI-facing LLM that never talks to a provider directly.

    All generation goes through LLMManager, which owns provider selection
    and fallback. Agents configured with this class have no visibility into
    which provider actually served a given call.
    """

    def __init__(self, manager: LLMManager, model: str = "rootcause-managed", **kwargs: Any):
        super().__init__(model=model, **kwargs)
        self.manager = manager
        self.last_provider: str | None = None

    def call(
        self,
        messages: str | list[dict],
        tools: list[dict] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        # CrewAI's ReAct tool-use loop relies on the model stopping right after
        # "Action Input: ..." so CrewAI can inject the real tool result and continue.
        # Without forwarding this stop sequence, a capable model will happily keep
        # generating past that point and fabricate a plausible-looking fake
        # "Observation" instead of ever letting the real tool run -- silently
        # replacing genuine search results with hallucinated ones.
        result = self.manager.generate(list(messages), stop=self.stop_sequences or None)
        self.last_provider = result.provider
        return result.content

    def supports_function_calling(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8192
