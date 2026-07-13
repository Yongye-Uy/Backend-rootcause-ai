from fastapi import APIRouter, HTTPException, Query

from app.llm.manager import AllProvidersFailedError, LLMManager, get_llm_manager

router = APIRouter(prefix="/api/debug", tags=["debug"])


@router.get("/llm-test")
def llm_test(
    prompt: str = Query(default="Reply with just the word OK."),
    exclude: str = Query(default="", description="Comma-separated provider names to skip, e.g. nvidia,gemini"),
) -> dict:
    """Calls LLMManager directly (bypassing CrewAI) to verify provider fallback."""
    manager = get_llm_manager()
    excluded = {name.strip() for name in exclude.split(",") if name.strip()}
    order = [name for name in manager.order if name not in excluded]

    test_manager = LLMManager(providers=manager.providers, order=order)
    try:
        result = test_manager.generate([{"role": "user", "content": prompt}])
    except AllProvidersFailedError as e:
        raise HTTPException(status_code=502, detail={"order_tried": order, "attempts": e.attempts}) from e

    return {"content": result.content, "provider": result.provider, "order_tried": order}
