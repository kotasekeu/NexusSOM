from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.database import get_db
from app.api.models import SomRun
from app.api.schemas import ImageSchema

router = APIRouter(prefix="/runs", tags=["images"])

_CATEGORY_RULES = [
    ("topology", lambda n: n.startswith("topology_")),
    ("dim_qe",   lambda n: n.startswith("dim_qe_")),
    ("dim",      lambda n: n.startswith("component_")),
    ("map",      lambda _: True),                       # fallback
]


def _classify(stem: str) -> str:
    for category, pred in _CATEGORY_RULES:
        if pred(stem):
            return category
    return "other"


@router.get("/{run_id}/images", response_model=list[ImageSchema])
def list_images(run_id: str, db: Session = Depends(get_db)):
    run = db.get(SomRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    run_path = Path(run.run_path)
    if not run_path.exists():
        return []

    images = []
    search_dirs = [
        (run_path, ""),
        (run_path / "visualizations", ""),
        (run_path / "maps_dataset", ""),
    ]
    for directory, _ in search_dirs:
        if directory.exists():
            for png in sorted(directory.glob("*.png")):
                rel = png.relative_to(run_path.parent.parent.parent)  # relative to data root
                images.append(ImageSchema(
                    name=png.stem,
                    category=_classify(png.stem),
                    path=f"/static/{rel}",
                ))

    return images
