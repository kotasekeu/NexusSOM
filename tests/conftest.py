"""
Shared fixtures for all NexusSom API tests.

Hierarchy:
  engine → db → client
  db + minimal_som_run → SOM integration tests
  db + minimal_ea_run  → EA integration tests
"""
import json
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from app.api.database import Base, get_db, init_db, reset_db
from app.api.main import app
from app.api.models import (
    Dataset, SomRun, SampleAssignment, NeuronQe, Cluster, Anomaly,
    EaRun, EaSeed, EaIndividual, EaParetoMetrics,
)


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

def _make_test_engine():
    """In-memory SQLite with StaticPool (single shared connection) and FK enforcement."""
    e = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(e, "connect")
    def _set_fk_pragma(conn, _):
        conn.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(e)
    return e


@pytest.fixture(scope="function")
def engine():
    e = _make_test_engine()
    yield e
    Base.metadata.drop_all(e)
    reset_db()


@pytest.fixture(scope="function")
def db(engine) -> Session:
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()


@pytest.fixture(scope="function")
def client(engine):
    """TestClient with in-memory DB injected via dependency override.

    init_db(engine) is called BEFORE TestClient so that when the lifespan
    calls init_db() with no args, it sees _engine already set and skips.
    """
    init_db(engine)         # registers engine globally (for /health) + skips in lifespan

    SessionLocal = sessionmaker(bind=engine)

    def override_get_db():
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    reset_db()


# ---------------------------------------------------------------------------
# Minimal SOM run fixture
# 3×3 map, 10 samples, 2 dims (x, y), 2 dead neurons, 2 outliers
# ---------------------------------------------------------------------------

SOM_RUN_ID = "test_run_001"
DATASET_NAME = "TestDataset"


@pytest.fixture(scope="function")
def minimal_som_run(db) -> str:
    """Inserts minimal SOM data, returns run_id."""
    ds = Dataset(name=DATASET_NAME, path="/fake/path", n_samples=10, n_dims=2, description="Test dataset")
    db.add(ds)
    db.flush()

    run = SomRun(
        id=SOM_RUN_ID, dataset_id=ds.id,
        map_m=3, map_n=3, mqe=0.15, topographic_error=0.05,
        dead_neuron_ratio=2/9, duration_s=10.0,
        run_path="/fake/path/results/test_run_001",
    )
    db.add(run)
    db.flush()

    # 7 active neurons (keys), 2 dead (0_2 and 2_0)
    active_keys = ["0_0", "0_1", "1_0", "1_1", "1_2", "2_1", "2_2"]
    assignments = [
        # sample_id, bmu_i, bmu_j, qe, dims, is_outlier
        (1,  0, 0, 0.10, {"x": 0.08, "y": 0.05}, 0),
        (2,  0, 1, 0.12, {"x": 0.10, "y": 0.06}, 0),
        (3,  1, 0, 0.09, {"x": 0.07, "y": 0.04}, 0),
        (4,  1, 1, 0.14, {"x": 0.11, "y": 0.08}, 0),
        (5,  1, 1, 0.13, {"x": 0.10, "y": 0.07}, 0),
        (6,  1, 2, 0.11, {"x": 0.09, "y": 0.06}, 0),
        (7,  2, 1, 0.08, {"x": 0.06, "y": 0.04}, 0),
        (8,  2, 2, 0.07, {"x": 0.05, "y": 0.03}, 0),
        (9,  0, 0, 0.45, {"x": 0.40, "y": 0.22}, 1),  # outlier
        (10, 2, 2, 0.52, {"x": 0.48, "y": 0.25}, 1),  # outlier
    ]
    for sid, bi, bj, qe, dims, outlier in assignments:
        db.add(SampleAssignment(
            run_id=SOM_RUN_ID, sample_id=sid,
            bmu_i=bi, bmu_j=bj, bmu_key=f"{bi}_{bj}",
            qe=qe, qe_dims=json.dumps(dims), is_outlier=outlier,
        ))

    # Clusters (7 active neurons)
    cluster_data = {
        "0_0": [1, 9], "0_1": [2], "1_0": [3], "1_1": [4, 5],
        "1_2": [6], "2_1": [7], "2_2": [8, 10],
    }
    for key, sample_ids in cluster_data.items():
        db.add(Cluster(run_id=SOM_RUN_ID, neuron_key=key, sample_ids=json.dumps(sample_ids), sample_count=len(sample_ids)))

    # NeuronQe
    neuron_qes = {
        "0_0": (0.275, 0.45), "0_1": (0.12, 0.12), "1_0": (0.09, 0.09),
        "1_1": (0.135, 0.14), "1_2": (0.11, 0.11), "2_1": (0.08, 0.08), "2_2": (0.295, 0.52),
    }
    for key, (mean, max_) in neuron_qes.items():
        db.add(NeuronQe(run_id=SOM_RUN_ID, neuron_key=key, qe_mean=mean, qe_max=max_,
                        sample_count=len(cluster_data[key])))

    # Anomalies
    db.add(Anomaly(run_id=SOM_RUN_ID, sample_id=9,  qe=0.45, reason=json.dumps([{"dim": "x", "type": "global_max"}])))
    db.add(Anomaly(run_id=SOM_RUN_ID, sample_id=10, qe=0.52, reason=json.dumps([{"dim": "y", "type": "global_max"}])))

    db.commit()
    return SOM_RUN_ID


# ---------------------------------------------------------------------------
# Minimal EA run fixture
# 1 seed, 3 generations, 5 individuals, 2 pareto_final
# ---------------------------------------------------------------------------

EA_RUN_ID = "test_ea_001"


@pytest.fixture(scope="function")
def minimal_ea_run(db) -> str:
    """Inserts minimal EA data, returns ea_run_id."""
    ds = db.query(Dataset).filter(Dataset.name == DATASET_NAME).first()
    if not ds:
        ds = Dataset(name=DATASET_NAME, path="/fake/path", n_samples=10, n_dims=2)
        db.add(ds)
        db.flush()

    ea_run = EaRun(id=EA_RUN_ID, dataset_id=ds.id, run_path="/fake/path/results/test_ea_001",
                   created_at="2026-06-01T10:00:00")
    db.add(ea_run)
    db.flush()

    seed = EaSeed(ea_run_id=EA_RUN_ID, seed_value=42, n_generations=3, final_hv=0.65, pareto_size=2)
    db.add(seed)
    db.flush()

    # 5 individuals across 3 generations
    individuals = [
        ("uid_a1", 1, 10, 10, 0.8, 0.06, False),
        ("uid_a2", 1, 12, 12, 0.7, 0.03, False),
        ("uid_b1", 2, 10, 10, 0.75, 0.05, False),
        ("uid_b2", 2, 12, 12, 0.65, 0.02, True),   # pareto_final
        ("uid_c1", 3, 10, 10, 0.60, 0.01, True),   # pareto_final
    ]
    for uid, gen, m, n, mqe, te, is_final in individuals:
        db.add(EaIndividual(
            uid=uid, seed_id=seed.id, generation=gen,
            map_m=m, map_n=n, mqe=mqe, mqe_ratio=mqe / 2,
            topographic_error=te, dead_ratio=0.0,
            is_pareto_final=int(is_final),
            hyperparams=json.dumps({"lr": 0.5, "radius": 3}),
        ))

    # Pareto metrics — HV must be non-decreasing
    for gen, hv in [(1, 0.30), (2, 0.50), (3, 0.65)]:
        db.add(EaParetoMetrics(seed_id=seed.id, generation=gen, front_size=gen, hv=hv,
                               spacing=0.1, spread_mqe=1.0, spread_te=1.0))

    db.commit()
    return EA_RUN_ID
