from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from sqlalchemy import desc
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as DBSession

from app import models
from app.agents import investigator, planner, research
from app.agents.planner import SelectedSolution
from app.database import SessionLocal, get_db
from app.guardrail import HEALTH_REFUSAL_MESSAGE, check_health_related
from app.llm.manager import get_llm_manager
from app.routers._shared import get_session_or_404 as _get_session_or_404
from app.schemas import (
    AnswersRequest,
    ConfirmRootCauseRequest,
    CreateSessionRequest,
    PlanOut,
    QAPairOut,
    SelectSolutionRequest,
    SessionStateResponse,
    SessionSummaryOut,
    SolutionOut,
    SourceOut,
)

ClientIdHeader = Header(..., alias="X-Client-Id")

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

MAX_CLARIFICATION_ROUNDS = 2
MAX_ROOT_CAUSE_REJECTIONS = 2


def _serialize(session: models.Session, message: str | None = None) -> SessionStateResponse:
    root_cause = max(session.root_causes, key=lambda rc: rc.id) if session.root_causes else None
    return SessionStateResponse(
        id=session.id,
        phase=session.phase.value,
        problem_text=session.problem_text,
        qa_pairs=[
            QAPairOut(round=qa.round, question=qa.question, answer=qa.answer)
            for qa in sorted(session.qa_pairs, key=lambda q: (q.round, q.id))
        ],
        root_cause=root_cause.description if root_cause else None,
        root_cause_confirmed=root_cause.confirmed if root_cause else False,
        solutions=[
            SolutionOut(
                id=sol.id,
                rank=sol.rank,
                name=sol.name,
                explanation=sol.explanation,
                resources=sol.resources,
                cost=sol.cost,
                difficulty=sol.difficulty,
                time_estimate=sol.time_estimate,
                pros=sol.pros,
                cons=sol.cons,
                risks=sol.risks,
                sources=[SourceOut(**s) for s in sol.sources],
            )
            for sol in sorted(session.solutions, key=lambda s: s.rank)
        ],
        selected_solution_id=session.selected_solution_id,
        llm_provider_used=session.llm_provider_used,
        plans=[
            PlanOut(
                id=p.id,
                solution_id=p.solution_id,
                llm_provider=p.llm_provider,
                overview=p.overview,
                requirements=p.requirements,
                tools=p.tools,
                cost=p.cost,
                timeline=p.timeline,
                steps=p.steps,
                possible_problems=p.possible_problems,
                alternatives=p.alternatives,
                sources=[SourceOut(**s) for s in p.sources],
            )
            for p in session.plans
        ],
        message=message,
        processing_steps=list(session.processing_steps),
    )


def _record_provider(session: models.Session) -> None:
    """Stamp the session with whichever provider most recently served an LLM call.

    Reflects the LLM Manager's automatic fallback in the UI ("answered via: X")
    without agents themselves ever knowing which provider handled them.
    """
    provider = get_llm_manager().last_provider
    if provider:
        session.llm_provider_used = provider


def _answered_qa_history(session: models.Session) -> list[tuple[str, str]]:
    return [
        (qa.question, qa.answer)
        for qa in sorted(session.qa_pairs, key=lambda q: (q.round, q.id))
        if qa.answer is not None
    ]


def _upsert_root_cause(
    db: DBSession, session: models.Session, description: str, confirmed: bool = False
) -> models.RootCause:
    """Reuse the session's current unconfirmed root-cause row if one exists.

    A session should only ever have one "live" root-cause row while it cycles
    through proposal/rejection — otherwise each re-proposal would start a
    fresh row with rejection_count reset to 0, breaking the rejection cap.
    """
    existing = max(session.root_causes, key=lambda rc: rc.id) if session.root_causes else None
    if existing is not None and not existing.confirmed:
        existing.description = description
        existing.confirmed = confirmed
        return existing

    root_cause = models.RootCause(session_id=session.id, description=description, confirmed=confirmed)
    db.add(root_cause)
    session.root_causes.append(root_cause)
    return root_cause


def _run_research_bg(session_id: int, root_cause_description: str) -> None:
    db = SessionLocal()
    try:
        session = db.query(models.Session).get(session_id)
        if not session:
            return

        def _step_callback(step: str):
            steps = list(session.processing_steps)
            steps.append(step)
            session.processing_steps = steps
            db.commit()

        _step_callback("Starting background research...")

        solutions = research.generate_solutions(
            session.problem_text, root_cause_description, progress_callback=_step_callback
        )
        
        if not solutions:
            # Revert to root cause confirm if research fails
            session.phase = models.Phase.ROOT_CAUSE_CONFIRM
            db.commit()
            return
            
        for rank, item in enumerate(solutions, start=1):
            db.add(
                models.Solution(
                    session_id=session.id,
                    rank=rank,
                    name=item.name,
                    explanation=item.explanation,
                    resources=item.resources,
                    cost=item.cost,
                    difficulty=item.difficulty,
                    time_estimate=item.time_estimate,
                    pros=item.pros,
                    cons=item.cons,
                    risks=item.risks,
                    sources=[s.model_dump() for s in item.sources],
                )
            )
        session.phase = models.Phase.SOLUTION_SELECT
        db.commit()
    except Exception as e:
        session = db.query(models.Session).get(session_id)
        if session:
            session.phase = models.Phase.ROOT_CAUSE_CONFIRM
            db.commit()
        print(f"Background research failed: {e}")
    finally:
        db.close()


def _analyze_answers_bg(
    session_id: int,
    problem_text: str,
    qa_pairs: list[tuple[str, str]],
    allow_followup: bool,
    extra_context: str,
    next_round: int
) -> None:
    db = SessionLocal()
    try:
        session = db.query(models.Session).get(session_id)
        if not session:
            return

        def _step_callback(step: str):
            steps = list(session.processing_steps)
            steps.append(step)
            session.processing_steps = steps
            db.commit()

        analysis = investigator.analyze_answers(
            problem_text=problem_text,
            qa_pairs=qa_pairs,
            allow_followup=allow_followup,
            extra_context=extra_context,
            progress_callback=_step_callback,
        )

        _upsert_root_cause(db, session, analysis.root_cause)

        if analysis.questions:
            for question in analysis.questions:
                db.add(models.QAPair(session_id=session.id, round=next_round, question=question, answer=None))

        session.phase = models.Phase.ROOT_CAUSE_CONFIRM
        _record_provider(session)
        db.commit()
    except Exception as e:
        print(f"Background analysis failed: {e}")
    finally:
        db.close()


def _analyze_and_research_bg(
    session_id: int, problem_text: str, qa_pairs: list[tuple[str, str]], extra_context: str
) -> None:
    db = SessionLocal()
    try:
        session = db.query(models.Session).get(session_id)
        if not session: return
        def _step_callback(step: str):
            steps = list(session.processing_steps)
            steps.append(step)
            session.processing_steps = steps
            db.commit()
            
        analysis = investigator.analyze_answers(
            problem_text, qa_pairs, allow_followup=False, extra_context=extra_context, progress_callback=_step_callback
        )
        final_root_cause = _upsert_root_cause(db, session, analysis.root_cause, confirmed=True)
        db.commit()
        
        _step_callback("Starting background research...")
        solutions = research.generate_solutions(
            session.problem_text, final_root_cause.description, progress_callback=_step_callback
        )
        if not solutions:
            session.phase = models.Phase.ROOT_CAUSE_CONFIRM
            db.commit()
            return
        
        for rank, item in enumerate(solutions, start=1):
            db.add(
                models.Solution(
                    session_id=session.id,
                    rank=rank,
                    name=item.name,
                    explanation=item.explanation,
                    resources=item.resources,
                    cost=item.cost,
                    difficulty=item.difficulty,
                    time_estimate=item.time_estimate,
                    pros=item.pros,
                    cons=item.cons,
                    risks=item.risks,
                    sources=[s.model_dump() for s in item.sources],
                )
            )
        session.phase = models.Phase.SOLUTION_SELECT
        db.commit()
    except Exception as e:
        print(f"Background analyze and research failed: {e}")
    finally:
        db.close()


def _generate_plan_bg(session_id: int, solution_id: int) -> None:
    db = SessionLocal()
    try:
        session = db.query(models.Session).get(session_id)
        if not session:
            return
        solution = db.query(models.Solution).get(solution_id)
        if not solution:
            return
            
        def _step_callback(step: str):
            steps = list(session.processing_steps)
            steps.append(step)
            session.processing_steps = steps
            db.commit()

        root_cause = max(session.root_causes, key=lambda rc: rc.id)
        plan_output = planner.generate_plan(
            session.problem_text,
            root_cause.description,
            SelectedSolution(
                name=solution.name,
                explanation=solution.explanation,
                resources=solution.resources,
                cost=solution.cost,
                difficulty=solution.difficulty,
                time_estimate=solution.time_estimate,
                sources=solution.sources,
            ),
            progress_callback=_step_callback
        )
        
        if not plan_output.steps or not plan_output.sources:
            session.phase = models.Phase.SOLUTION_SELECT
            db.commit()
            return

        llm_provider = get_llm_manager().last_provider
        db.add(
            models.Plan(
                session_id=session.id,
                solution_id=solution.id,
                llm_provider=llm_provider,
                overview=plan_output.overview,
                requirements=plan_output.requirements,
                tools=plan_output.tools,
                cost=plan_output.cost,
                timeline=plan_output.timeline,
                steps=plan_output.steps,
                possible_problems=plan_output.possible_problems,
                alternatives=plan_output.alternatives,
                sources=[s.model_dump() for s in plan_output.sources],
            )
        )
        session.selected_solution_id = solution.id
        session.phase = models.Phase.DONE
        _record_provider(session)
        db.commit()
    except Exception as e:
        session = db.query(models.Session).get(session_id)
        if session:
            session.phase = models.Phase.SOLUTION_SELECT
            db.commit()
        print(f"Background planning failed: {e}")
    finally:
        db.close()


@router.get("", response_model=list[SessionSummaryOut])
def list_sessions(
    db: DBSession = Depends(get_db), client_id: str = ClientIdHeader
) -> list[SessionSummaryOut]:
    sessions = (
        db.query(models.Session)
        .filter(models.Session.client_id == client_id)
        .order_by(desc(models.Session.updated_at))
        .all()
    )
    return [
        SessionSummaryOut(
            id=s.id,
            problem_text=s.problem_text,
            phase=s.phase.value,
            created_at=s.created_at,
            updated_at=s.updated_at,
        )
        for s in sessions
    ]


@router.post("", response_model=SessionStateResponse)
def create_session(
    payload: CreateSessionRequest, 
    background_tasks: BackgroundTasks,
    db: DBSession = Depends(get_db)
) -> SessionStateResponse:
    guardrail_result = check_health_related(payload.problem_text)

    if guardrail_result.is_health_related:
        session = models.Session(
            problem_text=payload.problem_text,
            client_id=payload.client_id,
            phase=models.Phase.REJECTED_HEALTH,
        )
        db.add(session)
        _record_provider(session)
        db.commit()
        db.refresh(session)
        return _serialize(session, message=HEALTH_REFUSAL_MESSAGE)

    session = models.Session(
        problem_text=payload.problem_text, client_id=payload.client_id, phase=models.Phase.CLARIFYING
    )
    session.processing_steps = []
    db.add(session)
    db.flush()

    background_tasks.add_task(
        _analyze_answers_bg,
        session_id=session.id,
        problem_text=payload.problem_text,
        qa_pairs=[],
        allow_followup=True,
        extra_context="",
        next_round=1
    )

    db.commit()
    db.refresh(session)
    return _serialize(session, message="Analyzing problem...")


@router.post("/{session_id}/answers", response_model=SessionStateResponse)
def submit_answers(
    session_id: int,
    payload: AnswersRequest,
    background_tasks: BackgroundTasks,
    db: DBSession = Depends(get_db),
    client_id: str = ClientIdHeader,
) -> SessionStateResponse:
    session = _get_session_or_404(session_id, client_id, db)
    if session.phase != models.Phase.ROOT_CAUSE_CONFIRM:
        raise HTTPException(
            status_code=400, detail=f"session is in phase '{session.phase.value}', not root cause confirm"
        )

    current_round = max((qa.round for qa in session.qa_pairs), default=1)
    unanswered = sorted(
        (qa for qa in session.qa_pairs if qa.round == current_round and qa.answer is None),
        key=lambda q: q.id,
    )
    if len(payload.answers) != len(unanswered):
        raise HTTPException(
            status_code=400,
            detail=f"expected {len(unanswered)} answers for round {current_round}, got {len(payload.answers)}",
        )
    for qa, answer in zip(unanswered, payload.answers):
        qa.answer = answer
    db.flush()

    allow_followup = current_round < MAX_CLARIFICATION_ROUNDS
    
    session.phase = models.Phase.CLARIFYING
    session.processing_steps = []
    
    background_tasks.add_task(
        _analyze_answers_bg,
        session_id=session.id,
        problem_text=session.problem_text,
        qa_pairs=_answered_qa_history(session),
        allow_followup=allow_followup,
        extra_context="",
        next_round=current_round + 1
    )

    db.commit()
    db.refresh(session)
    return _serialize(session, message="Thanks for the extra info. Analyzing...")


@router.post("/{session_id}/confirm-root-cause", response_model=SessionStateResponse)
def confirm_root_cause(
    session_id: int,
    payload: ConfirmRootCauseRequest,
    background_tasks: BackgroundTasks,
    db: DBSession = Depends(get_db),
    client_id: str = ClientIdHeader,
) -> SessionStateResponse:
    session = _get_session_or_404(session_id, client_id, db)
    if session.phase != models.Phase.ROOT_CAUSE_CONFIRM:
        raise HTTPException(
            status_code=400,
            detail=f"session is in phase '{session.phase.value}', not awaiting root cause confirmation",
        )

    root_cause = max(session.root_causes, key=lambda rc: rc.id)

    if payload.confirmed:
        root_cause.confirmed = True
        session.phase = models.Phase.RESEARCHING
        session.processing_steps = []
        background_tasks.add_task(_run_research_bg, session.id, root_cause.description)
        _record_provider(session)
        db.commit()
        db.refresh(session)
        return _serialize(session, message="Root cause confirmed. Initiating research...")

    root_cause.rejection_count += 1
    extra_context = f'The user rejected this proposed root cause: "{root_cause.description}".'
    if payload.feedback:
        extra_context += f' They gave this feedback: "{payload.feedback}"'

    if root_cause.rejection_count < MAX_ROOT_CAUSE_REJECTIONS:
        next_round = max((qa.round for qa in session.qa_pairs), default=0) + 1
        
        session.phase = models.Phase.CLARIFYING
        session.processing_steps = []
        background_tasks.add_task(
            _analyze_answers_bg,
            session_id=session.id,
            problem_text=session.problem_text,
            qa_pairs=_answered_qa_history(session),
            allow_followup=True,
            extra_context=extra_context,
            next_round=next_round
        )
        db.commit()
        db.refresh(session)
        return _serialize(session, message="Got it — re-analyzing...")

    session.phase = models.Phase.RESEARCHING
    session.processing_steps = []
    background_tasks.add_task(
        _analyze_and_research_bg,
        session_id=session.id,
        problem_text=session.problem_text,
        qa_pairs=_answered_qa_history(session),
        extra_context=extra_context
    )
    _record_provider(session)
    db.commit()
    db.refresh(session)
    return _serialize(
        session,
        message=(
            "We've reached the maximum number of attempts, so this root cause was accepted "
            "automatically. Initiating research..."
        ),
    )


@router.post("/{session_id}/select-solution", response_model=SessionStateResponse)
def select_solution(
    session_id: int,
    payload: SelectSolutionRequest,
    background_tasks: BackgroundTasks,
    db: DBSession = Depends(get_db),
    client_id: str = ClientIdHeader,
) -> SessionStateResponse:
    session = _get_session_or_404(session_id, client_id, db)
    if session.phase != models.Phase.SOLUTION_SELECT:
        raise HTTPException(
            status_code=400,
            detail=f"session is in phase '{session.phase.value}', not awaiting solution selection",
        )

    solution = next((s for s in session.solutions if s.id == payload.solution_id), None)
    if solution is None:
        raise HTTPException(status_code=404, detail="solution not found for this session")

    existing_plan = next((p for p in session.plans if p.solution_id == solution.id), None)
    if existing_plan is not None:
        # Cache hit: a plan for this solution already exists from an earlier visit.
        # No Planner Agent call, no provider stamp update -- nothing was generated.
        session.selected_solution_id = solution.id
        session.phase = models.Phase.DONE
        db.commit()
        db.refresh(session)
        return _serialize(session, message=f'"{solution.name}" selected. Showing your saved plan.')

    session.selected_solution_id = solution.id
    session.phase = models.Phase.PLANNING
    session.processing_steps = []
    
    background_tasks.add_task(_generate_plan_bg, session.id, solution.id)
    
    db.commit()
    db.refresh(session)
    return _serialize(session, message=f'"{solution.name}" selected. Generating plan...')


@router.post("/{session_id}/back-to-solutions", response_model=SessionStateResponse)
def back_to_solutions(
    session_id: int, db: DBSession = Depends(get_db), client_id: str = ClientIdHeader
) -> SessionStateResponse:
    session = _get_session_or_404(session_id, client_id, db)
    if session.phase != models.Phase.DONE:
        raise HTTPException(
            status_code=400,
            detail=f"session is in phase '{session.phase.value}', not done",
        )

    session.phase = models.Phase.SOLUTION_SELECT
    db.commit()
    db.refresh(session)
    return _serialize(session)


@router.get("/{session_id}", response_model=SessionStateResponse)
def get_session(
    session_id: int, db: DBSession = Depends(get_db), client_id: str = ClientIdHeader
) -> SessionStateResponse:
    session = _get_session_or_404(session_id, client_id, db)
    return _serialize(session)
