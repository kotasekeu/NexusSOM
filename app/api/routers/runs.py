import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.database import get_db
from app.api.models import SomRun, Dataset, Cluster, Anomaly, NeuronQe, SampleAssignment
from app.api.schemas import (
    RunSummarySchema, SomRunSchema, SomRunCreateSchema,
    ClusterBulkItem, SampleAssignmentBulkItem, AnomalyBulkItem,
    AnomalySchema, ClusterSchema,
)

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("", response_model=list[SomRunSchema])
def list_runs(db: Session = Depends(get_db)):
    return db.query(SomRun).order_by(SomRun.created_at.desc()).all()


@router.post("", response_model=SomRunSchema, status_code=status.HTTP_201_CREATED,
             summary="Register a SOM run",
             description="Creates a SOM run record. Dataset must exist. "
                         "Returns 409 if run ID already exists. "
                         "Use the bulk sub-endpoints to add clusters, assignments, and anomalies.")
def create_run(body: SomRunCreateSchema, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.name == body.dataset_name).first()
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{body.dataset_name}' not found")
    if db.get(SomRun, body.id):
        raise HTTPException(status_code=409, detail=f"Run '{body.id}' already exists")
    run = SomRun(
        id=body.id, dataset_id=ds.id, ea_uid=body.ea_uid,
        map_m=body.map_m, map_n=body.map_n,
        mqe=body.mqe, topographic_error=body.topographic_error,
        dead_neuron_ratio=body.dead_neuron_ratio, duration_s=body.duration_s,
        run_path=body.run_path, created_at=datetime.utcnow().isoformat(),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


@router.post("/{run_id}/clusters", status_code=status.HTTP_201_CREATED,
             summary="Bulk add clusters",
             description="Insert or replace cluster assignments for a run. "
                         "Existing clusters for the same neuron_key are deleted first (upsert by key).")
def bulk_add_clusters(run_id: str, items: list[ClusterBulkItem], db: Session = Depends(get_db)):
    if not db.get(SomRun, run_id):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    keys = [i.neuron_key for i in items]
    db.query(Cluster).filter(Cluster.run_id == run_id, Cluster.neuron_key.in_(keys)).delete()

    for item in items:
        db.add(Cluster(
            run_id=run_id, neuron_key=item.neuron_key,
            sample_ids=json.dumps(item.sample_ids), sample_count=len(item.sample_ids),
        ))

    # Recompute neuron_qe from existing sample_assignments
    _recompute_neuron_qe(run_id, db)
    db.commit()
    return {"inserted": len(items)}


@router.post("/{run_id}/assignments", status_code=status.HTTP_201_CREATED,
             summary="Bulk add sample assignments",
             description="Insert sample-to-neuron assignments. Appends to existing data. "
                         "Call `/clusters` endpoint afterwards to refresh cluster counts.")
def bulk_add_assignments(run_id: str, items: list[SampleAssignmentBulkItem], db: Session = Depends(get_db)):
    if not db.get(SomRun, run_id):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    for item in items:
        db.add(SampleAssignment(
            run_id=run_id, sample_id=item.sample_id,
            bmu_i=item.bmu_i, bmu_j=item.bmu_j, bmu_key=f"{item.bmu_i}_{item.bmu_j}",
            qe=item.qe,
            qe_dims=json.dumps(item.qe_dims) if item.qe_dims else None,
            is_outlier=int(item.is_outlier),
        ))
    db.commit()
    return {"inserted": len(items)}


@router.post("/{run_id}/anomalies", status_code=status.HTTP_201_CREATED,
             summary="Bulk add anomalies",
             description="Insert anomaly records. Appends to existing anomalies.")
def bulk_add_anomalies(run_id: str, items: list[AnomalyBulkItem], db: Session = Depends(get_db)):
    if not db.get(SomRun, run_id):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    for item in items:
        db.add(Anomaly(
            run_id=run_id, sample_id=item.sample_id, qe=item.qe,
            reason=json.dumps(item.reason) if item.reason is not None else None,
        ))
    db.commit()
    return {"inserted": len(items)}


@router.delete("/{run_id}", status_code=status.HTTP_204_NO_CONTENT,
               summary="Delete a SOM run and all its data")
def delete_run(run_id: str, db: Session = Depends(get_db)):
    run = db.get(SomRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    db.query(SampleAssignment).filter_by(run_id=run_id).delete()
    db.query(NeuronQe).filter_by(run_id=run_id).delete()
    db.query(Cluster).filter_by(run_id=run_id).delete()
    db.query(Anomaly).filter_by(run_id=run_id).delete()
    db.delete(run)
    db.commit()


def _recompute_neuron_qe(run_id: str, db: Session):
    """Recompute neuron_qe aggregates from sample_assignments after cluster/assignment changes."""
    db.query(NeuronQe).filter_by(run_id=run_id).delete()
    rows = db.query(SampleAssignment).filter_by(run_id=run_id).all()
    from collections import defaultdict
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.qe is not None:
            buckets[r.bmu_key].append(r.qe)
    for key, vals in buckets.items():
        db.add(NeuronQe(run_id=run_id, neuron_key=key,
                        qe_mean=sum(vals) / len(vals), qe_max=max(vals),
                        sample_count=len(vals)))


@router.get("/compare")
def compare_runs(ids: str = Query(..., description="Comma-separated run IDs"), db: Session = Depends(get_db)):
    run_ids = [i.strip() for i in ids.split(",") if i.strip()]
    if len(run_ids) < 2:
        raise HTTPException(status_code=422, detail="Provide at least 2 run IDs")
    runs = db.query(SomRun).filter(SomRun.id.in_(run_ids)).all()
    found_ids = {r.id for r in runs}
    missing = [rid for rid in run_ids if rid not in found_ids]
    if missing:
        raise HTTPException(status_code=404, detail=f"Runs not found: {missing}")
    return [SomRunSchema.model_validate(r) for r in runs]


@router.get("/{run_id}", response_model=SomRunSchema)
def get_run(run_id: str, db: Session = Depends(get_db)):
    run = db.get(SomRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return run


@router.get("/{run_id}/summary", response_model=RunSummarySchema,
            summary="Compact run overview for LLM init context",
            description="Returns key metrics and counts in a compact format. "
                        "Designed to fit within ~500 tokens — suitable as LLM init context.")
def get_run_summary(run_id: str, db: Session = Depends(get_db)):
    run = db.get(SomRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    dataset = db.get(Dataset, run.dataset_id)
    n_active = db.query(Cluster).filter(Cluster.run_id == run_id).count()
    n_anomalies = db.query(Anomaly).filter(Anomaly.run_id == run_id).count()

    total_neurons = run.map_m * run.map_n
    n_dead = total_neurons - n_active

    return RunSummarySchema(
        run_id=run_id,
        dataset=dataset.name if dataset else "",
        map_size=[run.map_m, run.map_n],
        mqe=run.mqe,
        topographic_error=run.topographic_error,
        n_samples=dataset.n_samples if dataset else None,
        n_dims=dataset.n_dims if dataset else None,
        n_clusters_active=n_active,
        n_dead_neurons=max(n_dead, 0),
        n_anomalies=n_anomalies,
        description=dataset.description if dataset else None,
    )
