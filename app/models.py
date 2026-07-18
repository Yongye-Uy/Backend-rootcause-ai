import enum
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Phase(str, enum.Enum):
    INTAKE = "intake"
    CLARIFYING = "clarifying"
    ROOT_CAUSE_CONFIRM = "root_cause_confirm"
    RESEARCHING = "researching"
    SOLUTION_SELECT = "solution_select"
    PLANNING = "planning"
    DONE = "done"
    REJECTED_HEALTH = "rejected_health"


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_id: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
    phase: Mapped[Phase] = mapped_column(Enum(Phase, name="phase"), default=Phase.INTAKE)
    problem_text: Mapped[str] = mapped_column(Text)
    llm_provider_used: Mapped[str | None] = mapped_column(String(50), nullable=True)
    selected_solution_id: Mapped[int | None] = mapped_column(
        ForeignKey("solutions.id"), nullable=True
    )
    processing_steps: Mapped[list[str]] = mapped_column(JSONB, default=list)

    qa_pairs: Mapped[list["QAPair"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", foreign_keys="QAPair.session_id"
    )
    root_causes: Mapped[list["RootCause"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        foreign_keys="RootCause.session_id",
    )
    solutions: Mapped[list["Solution"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        foreign_keys="Solution.session_id",
    )
    plans: Mapped[list["Plan"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        foreign_keys="Plan.session_id",
    )


class QAPair(Base):
    __tablename__ = "qa_pairs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"))
    round: Mapped[int] = mapped_column(Integer, default=1)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)

    session: Mapped["Session"] = relationship(back_populates="qa_pairs", foreign_keys=[session_id])


class RootCause(Base):
    __tablename__ = "root_causes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"))
    description: Mapped[str] = mapped_column(Text)
    confirmed: Mapped[bool] = mapped_column(default=False)
    rejection_count: Mapped[int] = mapped_column(Integer, default=0)

    session: Mapped["Session"] = relationship(
        back_populates="root_causes", foreign_keys=[session_id]
    )


class Solution(Base):
    __tablename__ = "solutions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"))
    rank: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(Text)
    explanation: Mapped[str] = mapped_column(Text)
    resources: Mapped[str] = mapped_column(Text)
    cost: Mapped[str] = mapped_column(Text)
    difficulty: Mapped[str] = mapped_column(Text)
    time_estimate: Mapped[str] = mapped_column(Text)
    pros: Mapped[list[str]] = mapped_column(JSONB, default=list)
    cons: Mapped[list[str]] = mapped_column(JSONB, default=list)
    risks: Mapped[list[str]] = mapped_column(JSONB, default=list)
    sources: Mapped[list[dict]] = mapped_column(JSONB, default=list)

    session: Mapped["Session"] = relationship(
        back_populates="solutions", foreign_keys=[session_id]
    )


class Plan(Base):
    __tablename__ = "plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), index=True)
    solution_id: Mapped[int] = mapped_column(ForeignKey("solutions.id"), unique=True, nullable=False)
    llm_provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    overview: Mapped[str] = mapped_column(Text)
    requirements: Mapped[str] = mapped_column(Text)
    tools: Mapped[str] = mapped_column(Text)
    cost: Mapped[str] = mapped_column(Text)
    timeline: Mapped[str] = mapped_column(Text)
    steps: Mapped[list[str]] = mapped_column(JSONB, default=list)
    possible_problems: Mapped[str] = mapped_column(Text)
    alternatives: Mapped[str] = mapped_column(Text)
    sources: Mapped[list[dict]] = mapped_column(JSONB, default=list)

    session: Mapped["Session"] = relationship(back_populates="plans", foreign_keys=[session_id])
