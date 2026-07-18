import re
from dataclasses import dataclass, field

from crewai import Agent, Crew, Task
from crewai.events import crewai_event_bus
from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent
from crewai_tools import TavilySearchTool

from app.agents.prompts import REACT_TOOL_FORMAT_REMINDER
from app.agents.schemas import PlanOutput
from app.llm.manager import get_llm_manager
from app.llm.managed_llm import ManagedLLM

_URL_RE = re.compile(r'https?://[^\s")\]]+')


@dataclass
class SelectedSolution:
    name: str
    explanation: str
    resources: str
    cost: str
    difficulty: str
    time_estimate: str
    sources: list[dict] = field(default_factory=list)  # [{"title": ..., "url": ...}, ...]


def _build_researcher() -> Agent:
    llm = ManagedLLM(get_llm_manager())
    return Agent(
        role="Implementation Researcher",
        goal="Find real, current, detailed how-to instructions for one specific solution using web search.",
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
        role="Implementation Planner",
        goal="Turn a chosen solution and research findings into a detailed, actionable step-by-step plan.",
        backstory=(
            "You are a meticulous project planner who only reports facts and sources that were "
            "actually given to you — you never invent a URL, and you always give concrete, "
            "practical estimates rather than vague hedges."
        ),
        llm=llm,
        memory=False,
        verbose=False,
    )


def _url_is_verified(url: str, seen_urls: set[str]) -> bool:
    # No fallback on an empty seen_urls: see the matching comment in agents/research.py.
    # A fabricated but well-formed URL must never pass just because nothing was captured.
    if url in seen_urls:
        return True
    return any(url in seen or seen in url for seen in seen_urls)


_PLACEHOLDER_DOMAIN_RE = re.compile(r"://(www\.)?example\.(com|org|net)(/|$|\?)", re.IGNORECASE)


def _is_well_formed_url(url: str) -> bool:
    if not re.match(r"^https?://[^\s]+\.[^\s]+", url):
        return False
    if _PLACEHOLDER_DOMAIN_RE.search(url):
        return False
    return True


MAX_ATTEMPTS = 2


from typing import Callable
from crewai.events.types.tool_usage_events import ToolUsageStartedEvent, ToolUsageFinishedEvent

def generate_plan(
    problem_text: str,
    root_cause: str,
    solution: SelectedSolution,
    progress_callback: Callable[[str], None] | None = None,
) -> PlanOutput:
    """Planner Agent occasionally has an off run (weak fallback provider, dropped tool call, etc.)
    and comes back with no steps or no verified sources. One retry is cheap insurance against that
    transient flakiness without risking an infinite loop."""
    if progress_callback:
        progress_callback("Planning research strategy...")

    plan = PlanOutput(
        overview="", requirements="", tools="", cost="", timeline="",
        possible_problems="", alternatives="",
    )
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if progress_callback and attempt > 1:
            progress_callback(f"Retrying plan generation (attempt {attempt})...")
            
        plan = _attempt_generate_plan(problem_text, root_cause, solution, progress_callback)
        if plan.steps and plan.sources:
            return plan
        if attempt < MAX_ATTEMPTS:
            print(f"Planner Agent returned an incomplete plan on attempt {attempt}, retrying")
    return plan


def _attempt_generate_plan(
    problem_text: str,
    root_cause: str,
    solution: SelectedSolution,
    progress_callback: Callable[[str], None] | None = None,
) -> PlanOutput:
    researcher = _build_researcher()
    writer = _build_writer()

    known_sources_text = "\n".join(f"- {s['title']}: {s['url']}" for s in solution.sources) or "(none)"

    search_task = Task(
        description=(
            f'Problem: """{problem_text}"""\n\n'
            f'Confirmed root cause: """{root_cause}"""\n\n'
            f'Chosen solution: "{solution.name}" — {solution.explanation}\n\n'
            f"Search the web for current, detailed, step-by-step instructions specifically for "
            f"carrying out this solution, plus any common pitfalls or troubleshooting tips. Run at "
            f"least 2 searches covering different angles (e.g. the step-by-step how-to, and common "
            f"mistakes/troubleshooting). Report your findings as plain text: for every useful "
            f"finding, include the exact title and URL exactly as returned by the search tool.\n\n"
            f"{REACT_TOOL_FORMAT_REMINDER}"
        ),
        expected_output=(
            "A plain-text summary of researched implementation details, each with the exact title "
            "and URL taken directly from the search tool's results."
        ),
        agent=researcher,
    )

    write_task = Task(
        description=(
            f"=== BACKGROUND CONTEXT (for your reference only — do not copy, list, or restate "
            f"these lines anywhere in your output) ===\n"
            f'Problem: """{problem_text}"""\n'
            f'Confirmed root cause: """{root_cause}"""\n'
            f'Chosen solution: "{solution.name}" — {solution.explanation}\n'
            f"Known resources: {solution.resources}\n"
            f"Known cost: {solution.cost} | difficulty: {solution.difficulty} | time: {solution.time_estimate}\n"
            f"Already-verified sources for this solution (safe to reuse/cite):\n{known_sources_text}\n"
            f"=== END BACKGROUND CONTEXT ===\n\n"
            f"Using the research findings you were given plus the background context above, write a "
            f"detailed implementation plan for carrying out the chosen solution. Every field must "
            f"contain genuinely new writing produced by you — never repeat the background context "
            f"lines verbatim or use them as list items.\n\n"
            f"Required fields:\n"
            f"- overview: 1-2 sentences summarizing the plan.\n"
            f"- requirements: what state the user/equipment must be in before starting.\n"
            f"- tools: a comma-separated list of physical tools/materials needed.\n"
            f"- cost: a concrete estimated total cost.\n"
            f"- timeline: a concrete estimated total time.\n"
            f"- steps: an ordered list of concrete PHYSICAL ACTIONS the user performs, e.g. "
            f'"Power off the laptop and unplug it", "Remove the four screws on the bottom panel". '
            f"Each step is one instruction, not a restatement of the problem/root cause/solution.\n"
            f"- possible_problems: things that could go wrong while doing the steps above, and how "
            f"to avoid or recognize them.\n"
            f"- alternatives: what to try if this solution doesn't fix the problem.\n"
            f"- sources: title + URL pairs.\n\n"
            f"Give concrete, practical estimates for cost/timeline — never 'not specified' or "
            f"'unknown'. The one constraint that matters is on SOURCES ONLY: only use source URLs "
            f"that literally appear in the research findings or the already-verified sources in the "
            f"background context — never invent, guess, or modify a URL."
        ),
        expected_output=(
            "A JSON object with overview, requirements, tools, cost, timeline, steps (list of "
            "concrete action strings, not a restatement of the input), possible_problems, "
            "alternatives, and sources, matching the required fields."
        ),
        agent=writer,
        context=[search_task],
        output_pydantic=PlanOutput,
    )

    crew = Crew(agents=[researcher, writer], tasks=[search_task, write_task], memory=False, verbose=False)

    seen_urls: set[str] = {s["url"] for s in solution.sources}
    
    def _tool_started(source, event: ToolUsageStartedEvent) -> None:
        if progress_callback:
            progress_callback(f"Using tool {event.tool_name} to find plan details...")

    def _capture_tool_output(source, event: ToolUsageFinishedEvent) -> None:
        if progress_callback:
            progress_callback("Analyzing gathered information...")
        if event.output:
            seen_urls.update(_URL_RE.findall(str(event.output)))

    with crewai_event_bus.scoped_handlers():
        crewai_event_bus.on(ToolUsageStartedEvent)(_tool_started)
        crewai_event_bus.on(ToolUsageFinishedEvent)(_capture_tool_output)
        
        if progress_callback:
            progress_callback("Starting research phase...")
            
        result = crew.kickoff()
        
        if progress_callback:
            progress_callback("Drafting final plan steps...")

    plan = result.pydantic if result.pydantic else PlanOutput(
        overview=result.raw, requirements="", tools="", cost="", timeline="",
        possible_problems="", alternatives="",
    )

    plan.sources = [
        s for s in plan.sources if _is_well_formed_url(s.url) and _url_is_verified(s.url, seen_urls)
    ]

    return plan
