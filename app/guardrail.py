import re
from dataclasses import dataclass

from app.llm.manager import LLMManager, get_llm_manager

CLASSIFIER_SYSTEM_PROMPT = """You are a strict, narrow classifier. You decide exactly one thing: \
is the user's problem fundamentally about diagnosing or treating a MEDICAL or MENTAL HEALTH \
condition of a HUMAN BEING (symptoms, illness, injury, disease, medication, mental health, etc.)?

Rules:
- Answer YES only if the core of the problem is a human health condition itself.
- Answer NO for technical, programming, business, career, relationship, learning, or daily-life \
problems, EVEN IF a health condition is mentioned as context (e.g. "my employee is often sick and \
it hurts our team's output" is a business problem, NOT a health problem to answer NO to).
- Answer NO for animal/pet health problems — this classifier only covers human health.
- When genuinely ambiguous, prefer NO (let the problem through) rather than over-refusing.

Respond with EXACTLY two lines, nothing else:
HEALTH: YES or HEALTH: NO
REASON: <one short sentence>
"""

_HEALTH_LINE_RE = re.compile(r"HEALTH:\s*(YES|NO)", re.IGNORECASE)
_REASON_LINE_RE = re.compile(r"REASON:\s*(.+)", re.IGNORECASE)


@dataclass
class GuardrailResult:
    is_health_related: bool
    reason: str
    provider: str
    raw: str


def check_health_related(problem_text: str, manager: LLMManager | None = None) -> GuardrailResult:
    manager = manager or get_llm_manager()
    messages = [
        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": problem_text},
    ]
    result = manager.generate(messages)

    health_match = _HEALTH_LINE_RE.search(result.content)
    reason_match = _REASON_LINE_RE.search(result.content)

    is_health_related = bool(health_match and health_match.group(1).upper() == "YES")
    reason = reason_match.group(1).strip() if reason_match else result.content.strip()

    return GuardrailResult(
        is_health_related=is_health_related,
        reason=reason,
        provider=result.provider,
        raw=result.content,
    )


HEALTH_REFUSAL_MESSAGE = (
    "This looks like a health-related problem. RootCause AI can't diagnose medical or mental "
    "health conditions or recommend treatments — please consult a qualified healthcare "
    "professional. I'm happy to help investigate a different (non-health) problem."
)
