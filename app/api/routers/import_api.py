import uuid
from fastapi import APIRouter
from app.api.schemas import ImportJobSchema

router = APIRouter(prefix="/import", tags=["import"])

_jobs: dict[str, dict] = {}


@router.post("/run", response_model=ImportJobSchema)
def trigger_import_run(run_path: str):
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "path": run_path}
    # TODO: launch background import task (Sprint 2)
    return ImportJobSchema(job_id=job_id, status="queued", message=f"Import queued for: {run_path}")


@router.get("/status/{job_id}", response_model=ImportJobSchema)
def get_import_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return ImportJobSchema(job_id=job_id, status=job["status"])
