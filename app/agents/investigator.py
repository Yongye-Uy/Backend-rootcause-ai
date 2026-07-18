from crewai import Agent, Crew, Task

from app.agents.schemas import QuestionList, RootCauseAnalysis
from app.llm.manager import get_llm_manager
from app.llm.managed_llm import ManagedLLM

DEFAULT_MAX_QUESTIONS = 5


def _build_agent() -> Agent:
    llm = ManagedLLM(get_llm_manager())
    return Agent(
        role="Problem Investigator",
        goal=(
            "Understand a user's problem deeply before anyone proposes solutions. Ask sharp, "
            "necessary clarification questions and identify the most likely root cause, never "
            "guessing when information is missing."
        ),
        backstory=(
            "You are a meticulous diagnostician who has learned that jumping to solutions before "
            "understanding a problem wastes everyone's time. You always investigate first."
        ),
        llm=llm,
        memory=False,
        verbose=False,
    )


from typing import Callable

def analyze_answers(
    problem_text: str,
    qa_pairs: list[tuple[str, str]],
    allow_followup: bool,
    extra_context: str = "",
    progress_callback: Callable[[str], None] | None = None,
) -> RootCauseAnalysis:
    if progress_callback:
        progress_callback("Analyzing problem context...")

    agent = _build_agent()
    qa_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in qa_pairs) if qa_pairs else "No clarification questions answered yet."

    if allow_followup:
        instruction = (
            "You must state your best-guess root cause based on the information available, even if imperfect. "
            "If you feel you need more information to be certain, you may optionally provide up to 3 targeted clarification questions."
        )
    else:
        instruction = (
            "You must now state your best-guess root cause based on the information available, "
            "even if imperfect. Do not ask any more clarification questions."
        )

    description = (
        f'Original problem:\n"""{problem_text}"""\n\n'
        f"Clarification so far:\n{qa_text}\n\n"
        f"{extra_context}\n\n"
        f"{instruction}"
    ).strip()
    task = Task(
        description=description,
        expected_output=(
            "A JSON object with 'questions' (list of strings, can be empty) and "
            "'root_cause' (string)."
        ),
        agent=agent,
        output_pydantic=RootCauseAnalysis,
    )
    
    if progress_callback:
        progress_callback("Generating root cause hypothesis...")
        
    crew = Crew(agents=[agent], tasks=[task], memory=False, verbose=False)
    result = crew.kickoff()
    
    if progress_callback:
        progress_callback("Finalizing analysis...")
        
    analysis = result.pydantic if result.pydantic else RootCauseAnalysis(root_cause=result.raw)

    if not allow_followup:
        analysis.questions = []

    return analysis

