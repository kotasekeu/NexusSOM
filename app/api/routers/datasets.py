from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.database import get_db
from app.api.models import Dataset, SomRun, EaRun
from app.api.schemas import DatasetSchema, DatasetCreateSchema, DatasetUpdateSchema, SomRunSchema, EaRunSchema

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("", response_model=list[DatasetSchema])
def list_datasets(db: Session = Depends(get_db)):
    return db.query(Dataset).order_by(Dataset.name).all()


@router.post("", response_model=DatasetSchema, status_code=status.HTTP_201_CREATED,
             summary="Register a dataset",
             description="Creates a new dataset entry. Returns 409 if a dataset with this name already exists.")
def create_dataset(body: DatasetCreateSchema, db: Session = Depends(get_db)):
    if db.query(Dataset).filter(Dataset.name == body.name).first():
        raise HTTPException(status_code=409, detail=f"Dataset '{body.name}' already exists")
    ds = Dataset(
        name=body.name, path=body.path, description=body.description,
        n_samples=body.n_samples, n_dims=body.n_dims, n_categorical=body.n_categorical,
        created_at=datetime.utcnow().isoformat(),
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return ds


@router.get("/{name}", response_model=DatasetSchema)
def get_dataset(name: str, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.name == name).first()
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    return ds


@router.put("/{name}", response_model=DatasetSchema,
            summary="Update dataset metadata",
            description="Updates description and/or sample/dimension counts. Only provided fields are changed.")
def update_dataset(name: str, body: DatasetUpdateSchema, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.name == name).first()
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    for field, value in body.model_dump(exclude_none=True).items():
        setattr(ds, field, value)
    db.commit()
    db.refresh(ds)
    return ds


@router.get("/{name}/runs")
def list_runs(name: str, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.name == name).first()
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")

    som_runs = db.query(SomRun).filter(SomRun.dataset_id == ds.id).order_by(SomRun.created_at.desc()).all()
    ea_runs = db.query(EaRun).filter(EaRun.dataset_id == ds.id).order_by(EaRun.created_at.desc()).all()

    return {
        "dataset": name,
        "som_runs": [SomRunSchema.model_validate(r) for r in som_runs],
        "ea_runs": [EaRunSchema.model_validate(r) for r in ea_runs],
    }
