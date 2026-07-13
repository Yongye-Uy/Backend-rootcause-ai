from dataclasses import dataclass, field
from datetime import datetime, timezone

from app import models


class ReportNotReadyError(Exception):
    pass


@dataclass
class Source:
    title: str
    url: str


@dataclass
class QARound:
    round: int
    pairs: list[tuple[str, str]]


@dataclass
class SolutionSummary:
    rank: int
    name: str
    cost: str
    difficulty: str
    time_estimate: str
    explanation: str


@dataclass
class SelectedSolutionDetail:
    name: str
    explanation: str
    resources: str
    cost: str
    difficulty: str
    time_estimate: str
    pros: list[str]
    cons: list[str]
    risks: list[str]
    sources: list[Source]


@dataclass
class PlanDetail:
    overview: str
    requirements: str
    tools: str
    cost: str
    timeline: str
    steps: list[str]
    possible_problems: str
    alternatives: str
    sources: list[Source]
    llm_provider: str | None


@dataclass
class ReportContext:
    session_id: int
    generated_at: datetime
    problem_text: str
    qa_rounds: list[QARound]
    root_cause: str
    root_cause_rejection_count: int
    solutions_summary: list[SolutionSummary]
    selected_solution: SelectedSolutionDetail
    plan: PlanDetail = field(repr=False)


def build_report_context(session: models.Session) -> ReportContext:
    if session.phase != models.Phase.DONE:
        raise ReportNotReadyError("session has no completed plan yet")

    plan_row = next((p for p in session.plans if p.solution_id == session.selected_solution_id), None)
    if plan_row is None:
        raise ReportNotReadyError("no plan found for the currently selected solution")

    solution_row = next((s for s in session.solutions if s.id == session.selected_solution_id), None)
    if solution_row is None:
        raise ReportNotReadyError("selected solution not found")

    root_cause_row = max(session.root_causes, key=lambda rc: rc.id) if session.root_causes else None

    rounds: dict[int, list[tuple[str, str]]] = {}
    for qa in sorted(session.qa_pairs, key=lambda q: (q.round, q.id)):
        rounds.setdefault(qa.round, []).append((qa.question, qa.answer or "(not answered)"))
    qa_rounds = [QARound(round=r, pairs=pairs) for r, pairs in sorted(rounds.items())]

    solutions_summary = [
        SolutionSummary(
            rank=sol.rank,
            name=sol.name,
            cost=sol.cost,
            difficulty=sol.difficulty,
            time_estimate=sol.time_estimate,
            explanation=sol.explanation,
        )
        for sol in sorted(session.solutions, key=lambda s: s.rank)
    ]

    selected_solution = SelectedSolutionDetail(
        name=solution_row.name,
        explanation=solution_row.explanation,
        resources=solution_row.resources,
        cost=solution_row.cost,
        difficulty=solution_row.difficulty,
        time_estimate=solution_row.time_estimate,
        pros=solution_row.pros,
        cons=solution_row.cons,
        risks=solution_row.risks,
        sources=[Source(**s) for s in solution_row.sources],
    )

    plan = PlanDetail(
        overview=plan_row.overview,
        requirements=plan_row.requirements,
        tools=plan_row.tools,
        cost=plan_row.cost,
        timeline=plan_row.timeline,
        steps=plan_row.steps,
        possible_problems=plan_row.possible_problems,
        alternatives=plan_row.alternatives,
        sources=[Source(**s) for s in plan_row.sources],
        llm_provider=plan_row.llm_provider,
    )

    return ReportContext(
        session_id=session.id,
        generated_at=datetime.now(timezone.utc),
        problem_text=session.problem_text,
        qa_rounds=qa_rounds,
        root_cause=root_cause_row.description if root_cause_row else "",
        root_cause_rejection_count=root_cause_row.rejection_count if root_cause_row else 0,
        solutions_summary=solutions_summary,
        selected_solution=selected_solution,
        plan=plan,
    )
