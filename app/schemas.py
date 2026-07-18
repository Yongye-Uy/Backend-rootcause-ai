from datetime import datetime

from pydantic import BaseModel


class CreateSessionRequest(BaseModel):
    problem_text: str
    client_id: str


class AnswersRequest(BaseModel):
    answers: list[str]


class ConfirmRootCauseRequest(BaseModel):
    confirmed: bool
    feedback: str | None = None


class SelectSolutionRequest(BaseModel):
    solution_id: int


class QAPairOut(BaseModel):
    round: int
    question: str
    answer: str | None


class SourceOut(BaseModel):
    title: str
    url: str


class SolutionOut(BaseModel):
    id: int
    rank: int
    name: str
    explanation: str
    resources: str
    cost: str
    difficulty: str
    time_estimate: str
    pros: list[str]
    cons: list[str]
    risks: list[str]
    sources: list[SourceOut]


class PlanOut(BaseModel):
    id: int
    solution_id: int
    llm_provider: str | None = None
    overview: str
    requirements: str
    tools: str
    cost: str
    timeline: str
    steps: list[str]
    possible_problems: str
    alternatives: str
    sources: list[SourceOut]


class SessionSummaryOut(BaseModel):
    id: int
    problem_text: str
    phase: str
    created_at: datetime
    updated_at: datetime


class SessionStateResponse(BaseModel):
    id: int
    phase: str
    problem_text: str
    qa_pairs: list[QAPairOut]
    root_cause: str | None = None
    root_cause_confirmed: bool = False
    solutions: list[SolutionOut] = []
    selected_solution_id: int | None = None
    plans: list[PlanOut] = []
    llm_provider_used: str | None = None
    message: str | None = None
    processing_steps: list[str] = []
