from fastapi import HTTPException
from sqlalchemy.orm import Session as DBSession

from app import models


def get_session_or_404(session_id: int, client_id: str, db: DBSession) -> models.Session:
    session = db.get(models.Session, session_id)
    if session is None or session.client_id != client_id:
        raise HTTPException(status_code=404, detail="session not found")
    return session
