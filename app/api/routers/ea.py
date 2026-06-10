import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.database import get_db
from app.api.models import EaRun, EaSeed, EaIndividual, EaParetoMetrics, Dataset
from app.api.schemas import (
    EaRunSchema, EaSeedSchema, EaIndividualSchema, EaParetoMetricsSchema,
    EaRunCreateSchema, EaSeedCreateSchema, EaIndividualBulkItem, EaParetoMetricsBulkItem,
)

router = APIRouter(prefix="/ea", tags=["ea"])


@router.get("", response_model=list[EaRunSchema])
def list_ea_runs(db: Session = Depends(get_db)):
    return db.query(EaRun).order_by(EaRun.created_at.desc()).all()


@router.post("", response_model=EaRunSchema, status_code=status.HTTP_201_CREATED,
             summary="Register an EA run")
def create_ea_run(body: EaRunCreateSchema, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.name == body.dataset_name).first()
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{body.dataset_name}' not found")
    if db.get(EaRun, body.id):
        raise HTTPException(status_code=409, detail=f"EA run '{body.id}' already exists")
    run = EaRun(id=body.id, dataset_id=ds.id, run_path=body.run_path,
                created_at=datetime.utcnow().isoformat())
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


@router.get("/{ea_run_id}")
def get_ea_run(ea_run_id: str, db: Session = Depends(get_db)):
    run = db.get(EaRun, ea_run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"EA run '{ea_run_id}' not found")

    dataset = db.get(Dataset, run.dataset_id)
    seeds = db.query(EaSeed).filter(EaSeed.ea_run_id == ea_run_id).all()

    return {
        "id": run.id,
        "dataset": dataset.name if dataset else None,
        "n_seeds": len(seeds),
        "seeds": [EaSeedSchema.model_validate(s) for s in seeds],
        "created_at": run.created_at,
    }


@router.get("/{ea_run_id}/seeds/{seed_id}/pareto")
def get_pareto_evolution(ea_run_id: str, seed_id: int, db: Session = Depends(get_db)):
    if not db.get(EaRun, ea_run_id):
        raise HTTPException(status_code=404, detail=f"EA run '{ea_run_id}' not found")

    seed = db.get(EaSeed, seed_id)
    if not seed or seed.ea_run_id != ea_run_id:
        raise HTTPException(status_code=404, detail=f"Seed {seed_id} not found in run '{ea_run_id}'")

    metrics = (
        db.query(EaParetoMetrics)
        .filter(EaParetoMetrics.seed_id == seed_id)
        .order_by(EaParetoMetrics.generation)
        .all()
    )
    pareto_final = (
        db.query(EaIndividual)
        .filter(EaIndividual.seed_id == seed_id, EaIndividual.is_pareto_final == 1)
        .all()
    )

    return {
        "seed_id": seed_id,
        "seed_value": seed.seed_value,
        "evolution": [EaParetoMetricsSchema.model_validate(m) for m in metrics],
        "pareto_final": [EaIndividualSchema.model_validate(i) for i in pareto_final],
    }


@router.post("/{ea_run_id}/seeds", response_model=EaSeedSchema, status_code=status.HTTP_201_CREATED,
             summary="Add a seed to an EA run")
def create_ea_seed(ea_run_id: str, body: EaSeedCreateSchema, db: Session = Depends(get_db)):
    if not db.get(EaRun, ea_run_id):
        raise HTTPException(status_code=404, detail=f"EA run '{ea_run_id}' not found")
    seed = EaSeed(ea_run_id=ea_run_id, **body.model_dump())
    db.add(seed)
    db.commit()
    db.refresh(seed)
    return seed


@router.post("/{ea_run_id}/seeds/{seed_id}/individuals", status_code=status.HTTP_201_CREATED,
             summary="Bulk add individuals to a seed",
             description="Insert EA individuals. Existing individuals with the same uid+seed_id are skipped (idempotent).")
def bulk_add_individuals(ea_run_id: str, seed_id: int, items: list[EaIndividualBulkItem],
                         db: Session = Depends(get_db)):
    if not db.get(EaRun, ea_run_id):
        raise HTTPException(status_code=404, detail=f"EA run '{ea_run_id}' not found")
    seed = db.get(EaSeed, seed_id)
    if not seed or seed.ea_run_id != ea_run_id:
        raise HTTPException(status_code=404, detail=f"Seed {seed_id} not found in run '{ea_run_id}'")

    existing = {r.uid for r in db.query(EaIndividual).filter_by(seed_id=seed_id).all()}
    inserted = 0
    for item in items:
        if item.uid in existing:
            continue
        db.add(EaIndividual(
            uid=item.uid, seed_id=seed_id, generation=item.generation,
            map_m=item.map_m, map_n=item.map_n,
            mqe=item.mqe, mqe_ratio=item.mqe_ratio,
            topographic_error=item.topographic_error, dead_ratio=item.dead_ratio,
            topo_corr=item.topo_corr, constraint_violation=item.constraint_violation,
            is_penalized=int(item.is_penalized), is_pareto_final=int(item.is_pareto_final),
            hyperparams=json.dumps(item.hyperparams) if item.hyperparams else None,
            duration_s=item.duration_s,
        ))
        inserted += 1
    db.commit()
    return {"inserted": inserted, "skipped": len(items) - inserted}


@router.post("/{ea_run_id}/seeds/{seed_id}/pareto_metrics", status_code=status.HTTP_201_CREATED,
             summary="Bulk add Pareto evolution metrics")
def bulk_add_pareto_metrics(ea_run_id: str, seed_id: int, items: list[EaParetoMetricsBulkItem],
                            db: Session = Depends(get_db)):
    if not db.get(EaRun, ea_run_id):
        raise HTTPException(status_code=404, detail=f"EA run '{ea_run_id}' not found")
    seed = db.get(EaSeed, seed_id)
    if not seed or seed.ea_run_id != ea_run_id:
        raise HTTPException(status_code=404, detail=f"Seed {seed_id} not found in run '{ea_run_id}'")

    existing_gens = {r.generation for r in db.query(EaParetoMetrics).filter_by(seed_id=seed_id).all()}
    inserted = 0
    for item in items:
        if item.generation in existing_gens:
            continue
        db.add(EaParetoMetrics(seed_id=seed_id, **item.model_dump()))
        inserted += 1
    db.commit()
    return {"inserted": inserted, "skipped": len(items) - inserted}


@router.delete("/{ea_run_id}", status_code=status.HTTP_204_NO_CONTENT,
               summary="Delete an EA run and all its data")
def delete_ea_run(ea_run_id: str, db: Session = Depends(get_db)):
    from app.api.models import CalibrationProbe
    run = db.get(EaRun, ea_run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"EA run '{ea_run_id}' not found")
    seed_ids = [s.id for s in db.query(EaSeed.id).filter_by(ea_run_id=ea_run_id)]
    db.query(EaIndividual).filter(EaIndividual.seed_id.in_(seed_ids)).delete()
    db.query(EaParetoMetrics).filter(EaParetoMetrics.seed_id.in_(seed_ids)).delete()
    db.query(EaSeed).filter_by(ea_run_id=ea_run_id).delete()
    db.query(CalibrationProbe).filter_by(ea_run_id=ea_run_id).delete()
    db.query(EaRun).filter_by(id=ea_run_id).delete()
    db.commit()


@router.get("/{ea_run_id}/individuals/{uid}", response_model=EaIndividualSchema)
def get_individual(ea_run_id: str, uid: str, db: Session = Depends(get_db)):
    if not db.get(EaRun, ea_run_id):
        raise HTTPException(status_code=404, detail=f"EA run '{ea_run_id}' not found")

    seeds = db.query(EaSeed).filter(EaSeed.ea_run_id == ea_run_id).all()
    seed_ids = {s.id for s in seeds}

    ind = (
        db.query(EaIndividual)
        .filter(EaIndividual.uid == uid, EaIndividual.seed_id.in_(seed_ids))
        .first()
    )
    if not ind:
        raise HTTPException(status_code=404, detail=f"Individual '{uid}' not found in EA run '{ea_run_id}'")
    return ind
