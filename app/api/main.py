from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import inspect

from app.api.config import DB_PATH, DATA_ROOT
from app.api.database import init_db, get_db
from app.api.routers import datasets, runs, clusters, anomalies, dimensions, images, ea, import_api


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="NexusSom API",
    version="2.0",
    description="REST API for NexusSom SOM + EA results",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if DATA_ROOT.exists():
    app.mount("/static", StaticFiles(directory=str(DATA_ROOT.parent)), name="static")

app.include_router(datasets.router)
app.include_router(runs.router)
app.include_router(clusters.router)
app.include_router(anomalies.router)
app.include_router(dimensions.router)
app.include_router(images.router)
app.include_router(ea.router)
app.include_router(import_api.router)


@app.get("/health", tags=["health"])
def health():
    from app.api.database import _engine
    from app.api.schemas import HealthSchema
    n_tables = len(inspect(_engine).get_table_names()) if _engine else 0
    return HealthSchema(status="ok", db_tables=n_tables)


if __name__ == "__main__":
    import uvicorn
    from app.api.config import API_HOST, API_PORT
    uvicorn.run("app.api.main:app", host=API_HOST, port=API_PORT, reload=True)
