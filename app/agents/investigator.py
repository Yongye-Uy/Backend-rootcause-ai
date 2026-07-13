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


def generate_questions(
    problem_text: str,
    max_questions: int = DEFAULT_MAX_QUESTIONS,
    extra_context: str = "",
) -> list[str]:
    agent = _build_agent()
    description = (
        f'The user described this problem:\n"""{problem_text}"""\n\n'
        f"{extra_context}\n\n"
        f"Identify the most important missing information needed to find the root cause. "
        f"Ask at most {max_questions} clear, specific clarification questions. "
        f"Do not propose any solutions or diagnoses yet — only ask questions."
    ).strip()
    task = Task(
        description=description,
        expected_output=(
            f"A JSON object with a 'questions' list of at most {max_questions} short, specific "
            f"questions."
        ),
        agent=agent,
        output_pydantic=QuestionList,
    )
    crew = Crew(agents=[agent], tasks=[task], memory=False, verbose=False)
    result = crew.kickoff()
    questions = result.pydantic.questions if result.pydantic else []
    return questions[:max_questions]


def analyze_answers(
    problem_text: str,
    qa_pairs: list[tuple[str, str]],
    allow_followup: bool,
    extra_context: str = "",
) -> RootCauseAnalysis:
    agent = _build_agent()
    qa_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in qa_pairs)

    if allow_followup:
        instruction = (
            "Decide if you have enough information to confidently state the root cause. "
            "If not, set needs_more_info=true and provide up to 3 more targeted questions "
            "(leave root_cause empty in that case). If you do have enough information, set "
            "needs_more_info=false and provide a clear, specific root_cause description (leave "
            "questions empty in that case)."
        )
    else:
        instruction = (
            "You must now state your best-guess root cause based on the information available, "
            "even if imperfect. Set needs_more_info=false and questions=[] no matter what, and "
            "provide your most likely root_cause description."
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
            "A JSON object with needs_more_info (bool), questions (list of strings), and "
            "root_cause (string)."
        ),
        agent=agent,
        output_pydantic=RootCauseAnalysis,
    )
    crew = Crew(agents=[agent], tasks=[task], memory=False, verbose=False)
    result = crew.kickoff()
    analysis = result.pydantic if result.pydantic else RootCauseAnalysis(root_cause=result.raw)

    if not allow_followup:
        analysis.needs_more_info = False
        analysis.questions = []

    return analysis
