import json
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.database import get_db
from app.api.models import SomRun, SampleAssignment
from app.api.schemas import DimensionStatsSchema

router = APIRouter(prefix="/runs", tags=["dimensions"])


def _aggregate_dims(rows: list[SampleAssignment]) -> dict[str, dict]:
    acc: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if not row.qe_dims:
            continue
        try:
            dims = json.loads(row.qe_dims)
        except (ValueError, TypeError):
            continue
        for name, val in dims.items():
            if val is not None:
                acc[name].append(float(val))

    result = {}
    for name, vals in acc.items():
        result[name] = {
            "qe_mean": sum(vals) / len(vals),
            "qe_max": max(vals),
        }
    return result


@router.get("/{run_id}/dimensions", response_model=list[DimensionStatsSchema])
def list_dimensions(run_id: str, db: Session = Depends(get_db)):
    if not db.get(SomRun, run_id):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    rows = db.query(SampleAssignment).filter(SampleAssignment.run_id == run_id).all()
    aggregated = _aggregate_dims(rows)

    return [DimensionStatsSchema(name=name, **stats) for name, stats in sorted(aggregated.items())]


@router.get("/{run_id}/dimensions/{name}", response_model=DimensionStatsSchema)
def get_dimension(run_id: str, name: str, db: Session = Depends(get_db)):
    if not db.get(SomRun, run_id):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    rows = db.query(SampleAssignment).filter(SampleAssignment.run_id == run_id).all()
    aggregated = _aggregate_dims(rows)

    if name not in aggregated:
        raise HTTPException(status_code=404, detail=f"Dimension '{name}' not found in run '{run_id}'")

    return DimensionStatsSchema(name=name, **aggregated[name])
