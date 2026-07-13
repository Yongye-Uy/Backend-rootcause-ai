from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session as DBSession

from app.database import get_db
from app.reports.data import ReportNotReadyError, build_report_context
from app.reports.pdf import generate_pdf_report
from app.routers._shared import get_session_or_404

router = APIRouter(prefix="/api/sessions", tags=["reports"])


@router.get("/{session_id}/report.pdf")
def download_pdf_report(session_id: int, client_id: str, db: DBSession = Depends(get_db)) -> Response:
    session = get_session_or_404(session_id, client_id, db)
    try:
        context = build_report_context(session)
    except ReportNotReadyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    pdf_bytes = generate_pdf_report(context)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="rootcause-ai-report-{session_id}.pdf"'
        },
    )
