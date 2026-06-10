from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.database import get_db
from app.api.models import SomRun, Cluster, NeuronQe
from app.api.schemas import ClusterSchema, ClusterDetailSchema

router = APIRouter(prefix="/runs", tags=["clusters"])

_VALID_SORT = {"sample_count", "qe_mean", "qe_max", "neuron_key"}


@router.get("/{run_id}/clusters", response_model=list[ClusterSchema],
            summary="List all active neurons (clusters)",
            description="Returns neurons that have at least 1 sample assigned. "
                        "Dead neurons (0 samples) are excluded. "
                        "Use `?sort_by=sample_count&top=10` to get the 10 largest clusters.")
def list_clusters(
    run_id: str,
    top: int = Query(default=0, ge=0, description="Return only top N results (0 = all)"),
    sort_by: str = Query(default="neuron_key", description="Sort by: neuron_key | sample_count | qe_mean | qe_max"),
    db: Session = Depends(get_db),
):
    if not db.get(SomRun, run_id):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    if sort_by not in _VALID_SORT:
        raise HTTPException(status_code=422, detail=f"Invalid sort_by '{sort_by}'. Valid: {sorted(_VALID_SORT)}")

    rows = (
        db.query(Cluster, NeuronQe)
        .outerjoin(NeuronQe, (NeuronQe.run_id == Cluster.run_id) & (NeuronQe.neuron_key == Cluster.neuron_key))
        .filter(Cluster.run_id == run_id)
        .all()
    )

    results = [
        ClusterSchema(
            neuron_key=c.neuron_key,
            sample_count=c.sample_count,
            qe_mean=nq.qe_mean if nq else None,
            qe_max=nq.qe_max if nq else None,
        )
        for c, nq in rows
    ]

    key_fn = {
        "sample_count": lambda x: (x.sample_count or 0),
        "qe_mean": lambda x: (x.qe_mean or 0),
        "qe_max": lambda x: (x.qe_max or 0),
        "neuron_key": lambda x: x.neuron_key,
    }[sort_by]

    reverse = sort_by in {"sample_count", "qe_mean", "qe_max"}
    results.sort(key=key_fn, reverse=reverse)

    return results[:top] if top > 0 else results


@router.get("/{run_id}/clusters/{neuron_key}", response_model=ClusterDetailSchema)
def get_cluster(run_id: str, neuron_key: str, db: Session = Depends(get_db)):
    if not db.get(SomRun, run_id):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    cluster = db.query(Cluster).filter(Cluster.run_id == run_id, Cluster.neuron_key == neuron_key).first()
    if not cluster:
        raise HTTPException(status_code=404, detail=f"Neuron '{neuron_key}' not found in run '{run_id}'")

    nq = db.query(NeuronQe).filter(NeuronQe.run_id == run_id, NeuronQe.neuron_key == neuron_key).first()

    return ClusterDetailSchema(
        neuron_key=cluster.neuron_key,
        sample_count=cluster.sample_count,
        qe_mean=nq.qe_mean if nq else None,
        qe_max=nq.qe_max if nq else None,
        sample_ids=cluster.sample_ids,
    )
