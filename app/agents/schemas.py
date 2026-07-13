from pydantic import BaseModel, Field


class QuestionList(BaseModel):
    questions: list[str] = Field(default_factory=list)


class RootCauseAnalysis(BaseModel):
    needs_more_info: bool
    questions: list[str] = Field(default_factory=list)
    root_cause: str = ""


class SourceItem(BaseModel):
    title: str
    url: str


class SolutionItem(BaseModel):
    name: str
    explanation: str
    resources: str
    cost: str
    difficulty: str
    time_estimate: str
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    sources: list[SourceItem] = Field(default_factory=list)


class SolutionList(BaseModel):
    solutions: list[SolutionItem] = Field(default_factory=list)


class PlanOutput(BaseModel):
    overview: str
    requirements: str
    tools: str
    cost: str
    timeline: str
    steps: list[str] = Field(default_factory=list)
    possible_problems: str
    alternatives: str
    sources: list[SourceItem] = Field(default_factory=list)
