from sqlalchemy import create_engine, Engine, event
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Session

from app.api.config import DB_PATH


class Base(DeclarativeBase):
    pass


def _make_file_engine(url: str) -> Engine:
    return create_engine(url, connect_args={"check_same_thread": False})


_engine: Engine | None = None
_SessionLocal = None


def init_db(engine_or_url: Engine | str | None = None):
    """Initialize DB.

    - Pass an Engine to use it directly (e.g. in tests with StaticPool).
    - Pass a URL string to create a new engine from it.
    - Pass nothing: init from DB_PATH, but skip if already initialized
      (so TestClient lifespan doesn't overwrite a test engine).
    """
    global _engine, _SessionLocal

    if engine_or_url is None:
        if _engine is not None:
            return _engine          # Already set by test fixture — skip
        engine_or_url = f"sqlite:///{DB_PATH}"

    if isinstance(engine_or_url, Engine):
        _engine = engine_or_url
    else:
        _engine = _make_file_engine(str(engine_or_url))

    _SessionLocal = sessionmaker(bind=_engine)
    Base.metadata.create_all(_engine)
    return _engine


def reset_db():
    """Reset global state (used in tests to allow re-initialization)."""
    global _engine, _SessionLocal
    _engine = None
    _SessionLocal = None


def get_db():
    """FastAPI dependency — yields a DB session."""
    if _SessionLocal is None:
        raise RuntimeError("DB not initialized — call init_db() first")
    db: Session = _SessionLocal()
    try:
        yield db
    finally:
        db.close()
