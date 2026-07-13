import re

from crewai import Agent, Crew, Task
from crewai.events import crewai_event_bus
from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent
from crewai_tools import TavilySearchTool

from app.agents.prompts import REACT_TOOL_FORMAT_REMINDER
from app.agents.schemas import SolutionItem, SolutionList
from app.llm.manager import get_llm_manager
from app.llm.managed_llm import ManagedLLM

DEFAULT_MAX_SOLUTIONS = 5

_URL_RE = re.compile(r'https?://[^\s")\]]+')


def _build_researcher() -> Agent:
    llm = ManagedLLM(get_llm_manager())
    return Agent(
        role="Solution Researcher",
        goal="Find real, current information about how to fix a confirmed root cause using web search.",
        backstory=(
            "You are a thorough researcher who always checks current sources before reporting "
            "anything, and always includes the exact title and URL of what you found."
        ),
        llm=llm,
        tools=[TavilySearchTool()],
        memory=False,
        verbose=False,
    )


def _build_writer() -> Agent:
    llm = ManagedLLM(get_llm_manager())
    return Agent(
        role="Solution Writer",
        goal="Turn research findings into a clear, structured, ranked list of candidate solutions.",
        backstory=(
            "You are a careful technical writer who only reports facts and sources that were "
            "actually given to you by the researcher — you never invent a URL."
        ),
        llm=llm,
        memory=False,
        verbose=False,
    )


def _url_is_verified(url: str, seen_urls: set[str]) -> bool:
    # No fallback here on purpose: if the search tool was never actually invoked (or its
    # output wasn't captured), seen_urls is empty and every source gets rejected rather
    # than trusted -- a fabricated but well-formed URL (e.g. https://example.com/...)
    # must never be presented to the user as real. The outer retry loop in
    # generate_solutions() already handles "came back with zero verified sources" by
    # retrying the whole attempt, so this fails safe instead of failing open.
    if url in seen_urls:
        return True
    return any(url in seen or seen in url for seen in seen_urls)


_PLACEHOLDER_DOMAIN_RE = re.compile(r"://(www\.)?example\.(com|org|net)(/|$|\?)", re.IGNORECASE)


def _is_well_formed_url(url: str) -> bool:
    if not re.match(r"^https?://[^\s]+\.[^\s]+", url):
        return False
    # example.com/.org/.net are RFC 2606 reserved documentation domains -- never a real
    # source, but a classic LLM hallucination placeholder.
    if _PLACEHOLDER_DOMAIN_RE.search(url):
        return False
    return True


MAX_ATTEMPTS = 2


def generate_solutions(
    problem_text: str, root_cause: str, max_solutions: int = DEFAULT_MAX_SOLUTIONS
) -> list[SolutionItem]:
    """Research Agent occasionally has an off run (weak fallback provider, dropped tool call,
    etc.) and comes back with zero usable solutions. One retry is cheap insurance against that
    transient flakiness without risking an infinite loop."""
    solutions: list[SolutionItem] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        solutions = _attempt_generate_solutions(problem_text, root_cause, max_solutions)
        if solutions:
            return solutions
        if attempt < MAX_ATTEMPTS:
            print(f"Research Agent returned 0 verified solutions on attempt {attempt}, retrying")
    return solutions


def _attempt_generate_solutions(
    problem_text: str, root_cause: str, max_solutions: int
) -> list[SolutionItem]:
    researcher = _build_researcher()
    writer = _build_writer()

    search_task = Task(
        description=(
            f'Problem: """{problem_text}"""\n\n'
            f'Confirmed root cause: """{root_cause}"""\n\n'
            f"Search the web for current, real fixes/solutions for this root cause. Run at least "
            f"2 searches covering different angles (e.g. quick DIY fixes vs professional repair vs "
            f"replacement/upgrade options). Report your findings as plain text: for every useful "
            f"finding, include the exact title and URL exactly as returned by the search tool.\n\n"
            f"{REACT_TOOL_FORMAT_REMINDER}"
        ),
        expected_output=(
            "A plain-text summary of researched solutions, each with the exact title and URL taken "
            "directly from the search tool's results."
        ),
        agent=researcher,
    )

    write_task = Task(
        description=(
            f'Problem: """{problem_text}"""\n\n'
            f'Confirmed root cause: """{root_cause}"""\n\n'
            f"Using the research findings you were given as your factual basis, produce the top "
            f"{max_solutions} distinct solutions, ranked best first. For each solution, provide: "
            f"name, explanation, resources needed, and at least one source (title + URL). "
            f"For cost, difficulty level, estimated time, advantages (pros), disadvantages (cons), "
            f"and risks: these are YOUR OWN practical, expert-judgment estimates for a typical user "
            f"attempting this solution (e.g. 'Free - $20 for compressed air', 'Easy, 15-30 minutes') "
            f"— give a concrete best estimate, never 'not specified' or 'unknown'. "
            f"The one constraint that matters is on SOURCES ONLY: only use source URLs that literally "
            f"appear in the research findings you were given — never invent, guess, or modify a URL. "
            f"If a solution has no matching source in the findings, do not include that solution."
        ),
        expected_output=(
            f"A JSON object with a 'solutions' list of up to {max_solutions} solutions, each matching "
            f"the required fields, each with at least one real source."
        ),
        agent=writer,
        context=[search_task],
        output_pydantic=SolutionList,
    )

    crew = Crew(agents=[researcher, writer], tasks=[search_task, write_task], memory=False, verbose=False)

    seen_urls: set[str] = set()

    def _capture_tool_output(source, event: ToolUsageFinishedEvent) -> None:
        if event.output:
            seen_urls.update(_URL_RE.findall(str(event.output)))

    with crewai_event_bus.scoped_handlers():
        crewai_event_bus.on(ToolUsageFinishedEvent)(_capture_tool_output)
        result = crew.kickoff()

    solutions = result.pydantic.solutions if result.pydantic else []

    verified_solutions = []
    for solution in solutions:
        solution.sources = [
            s
            for s in solution.sources
            if _is_well_formed_url(s.url) and _url_is_verified(s.url, seen_urls)
        ]
        if solution.sources:
            verified_solutions.append(solution)

    return verified_solutions[:max_solutions]
