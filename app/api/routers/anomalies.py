from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.database import get_db
from app.api.models import SomRun, Anomaly
from app.api.schemas import AnomalySchema

router = APIRouter(prefix="/runs", tags=["anomalies"])


@router.get("/{run_id}/anomalies", response_model=list[AnomalySchema])
def list_anomalies(
    run_id: str,
    limit: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    if not db.get(SomRun, run_id):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    q = db.query(Anomaly).filter(Anomaly.run_id == run_id).order_by(Anomaly.qe.desc())
    rows = q.limit(limit).all() if limit > 0 else q.all()
    return rows
