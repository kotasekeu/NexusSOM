import json
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DatasetSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, json_schema_extra={
        "example": {"id": 1, "name": "WineQuality", "n_samples": 1599, "n_dims": 11,
                    "n_categorical": 0, "description": "Red wine quality dataset.", "created_at": "2026-06-01T10:00:00"}
    })

    id: int
    name: str
    n_samples: Optional[int] = None
    n_dims: Optional[int] = None
    n_categorical: Optional[int] = None
    description: Optional[str] = None
    created_at: Optional[str] = None


class SomRunSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, json_schema_extra={
        "example": {"id": "20260530_125716", "dataset_id": 1, "map_m": 15, "map_n": 15,
                    "mqe": 0.1247, "topographic_error": 0.023, "dead_neuron_ratio": 0.09,
                    "duration_s": 752.8, "created_at": "2026-05-30T12:57:16"}
    })

    id: str = Field(description="Run timestamp ID, e.g. '20260530_125716'")
    dataset_id: int
    map_m: int = Field(description="Map rows")
    map_n: int = Field(description="Map columns")
    mqe: Optional[float] = Field(default=None, description="Mean Quantization Error")
    topographic_error: Optional[float] = Field(default=None, description="Topographic Error (0–1)")
    dead_neuron_ratio: Optional[float] = Field(default=None, description="Fraction of neurons with 0 samples")
    duration_s: Optional[float] = None
    created_at: Optional[str] = None


class RunSummarySchema(BaseModel):
    """Compact run overview — designed to fit within ~500 LLM tokens."""

    model_config = ConfigDict(json_schema_extra={"example": {
        "run_id": "20260530_125716", "dataset": "WineQuality", "map_size": [15, 15],
        "mqe": 0.1247, "topographic_error": 0.023, "n_samples": 1599, "n_dims": 11,
        "n_clusters_active": 180, "n_dead_neurons": 45, "n_anomalies": 12,
        "description": "Red wine quality dataset."
    }})

    run_id: str
    dataset: str
    map_size: list[int] = Field(description="[rows, cols]")
    mqe: Optional[float] = Field(default=None, description="Mean Quantization Error")
    topographic_error: Optional[float] = Field(default=None, description="Topographic Error (0–1)")
    n_samples: Optional[int] = None
    n_dims: Optional[int] = None
    n_clusters_active: int = Field(description="Neurons with at least 1 sample assigned")
    n_dead_neurons: int = Field(description="Neurons with 0 samples")
    n_anomalies: int
    description: Optional[str] = Field(default=None, description="Dataset description from ABOUT.md")


class ClusterSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, json_schema_extra={
        "example": {"neuron_key": "3_4", "sample_count": 12, "qe_mean": 0.082, "qe_max": 0.154}
    })

    neuron_key: str = Field(description="Neuron position as 'row_col', e.g. '3_4'")
    sample_count: Optional[int] = None
    qe_mean: Optional[float] = None
    qe_max: Optional[float] = None


class ClusterDetailSchema(ClusterSchema):
    model_config = ConfigDict(from_attributes=True, json_schema_extra={
        "example": {"neuron_key": "3_4", "sample_count": 3, "qe_mean": 0.082, "qe_max": 0.154,
                    "sample_ids": [42, 117, 389]}
    })

    sample_ids: list[int] = Field(default=[], description="IDs of samples assigned to this neuron")

    @field_validator("sample_ids", mode="before")
    @classmethod
    def parse_sample_ids(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v or []


class AnomalySchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, json_schema_extra={
        "example": {"sample_id": 42, "qe": 0.48, "reason": [{"dim": "alcohol", "type": "global_max", "value": 14.9}]}
    })

    sample_id: Optional[int] = None
    qe: Optional[float] = Field(default=None, description="Quantization error — higher = more anomalous")
    reason: Optional[Any] = Field(default=None, description="List of reason dicts from extremes.json")

    @field_validator("reason", mode="before")
    @classmethod
    def parse_reason(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (ValueError, TypeError):
                return v
        return v


class DimensionStatsSchema(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {"name": "alcohol", "qe_mean": 0.043, "qe_max": 0.412}
    })

    name: str = Field(description="Dimension / feature name")
    qe_mean: Optional[float] = Field(default=None, description="Mean per-dimension QE across all samples")
    qe_max: Optional[float] = Field(default=None, description="Max per-dimension QE across all samples")


class ImageSchema(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {"name": "u_matrix", "category": "map", "path": "/static/datasets/WineQuality/results/.../u_matrix.png"}
    })

    name: str
    category: str = Field(description="One of: map | dim | dim_qe | topology")
    path: str = Field(description="/static/... URL, serve directly from FastAPI static mount")


class EaRunSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, json_schema_extra={
        "example": {"id": "20260515_111812", "dataset_id": 2, "created_at": "2026-05-15T11:18:12"}
    })

    id: str
    dataset_id: Optional[int] = None
    created_at: Optional[str] = None


class EaSeedSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, json_schema_extra={
        "example": {"id": 1, "seed_value": 42, "n_generations": 50, "final_hv": 0.65, "pareto_size": 5}
    })

    id: int
    seed_value: Optional[int] = None
    n_generations: Optional[int] = None
    final_hv: Optional[float] = Field(default=None, description="Final hypervolume of Pareto front")
    pareto_size: Optional[int] = Field(default=None, description="Number of individuals on final Pareto front")


class EaIndividualSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, json_schema_extra={
        "example": {"uid": "30e740aa", "generation": 12, "map_m": 18, "map_n": 18,
                    "mqe": 0.666, "topographic_error": 0.021, "dead_ratio": 0.006,
                    "is_pareto_final": 1, "hyperparams": {"lr": 0.57, "radius_ratio": 0.935}}
    })

    uid: str
    generation: Optional[int] = None
    map_m: Optional[int] = None
    map_n: Optional[int] = None
    mqe: Optional[float] = None
    mqe_ratio: Optional[float] = Field(default=None, description="Normalized MQE (lower = better)")
    topographic_error: Optional[float] = None
    dead_ratio: Optional[float] = None
    topo_corr: Optional[float] = Field(default=None, description="Spearman ρ topological correlation")
    is_pareto_final: int = Field(default=0, description="1 = on final Pareto front")
    hyperparams: Optional[Any] = Field(default=None, description="SOM hyperparameters as dict")

    @field_validator("hyperparams", mode="before")
    @classmethod
    def parse_hyperparams(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (ValueError, TypeError):
                return v
        return v


class EaParetoMetricsSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, json_schema_extra={
        "example": {"generation": 10, "front_size": 5, "hv": 0.52, "spacing": 0.11,
                    "spread_mqe": 0.8, "spread_te": 0.6}
    })

    generation: int
    front_size: Optional[int] = None
    hv: Optional[float] = Field(default=None, description="Hypervolume indicator — higher is better")
    spacing: Optional[float] = None
    spread_mqe: Optional[float] = None
    spread_te: Optional[float] = None


# ---------------------------------------------------------------------------
# Request schemas (POST / PUT bodies)
# ---------------------------------------------------------------------------

class DatasetCreateSchema(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {"name": "WineQuality", "path": "data/datasets/WineQuality",
                    "description": "Red wine quality dataset.", "n_samples": 1599,
                    "n_dims": 11, "n_categorical": 0}
    })

    name: str = Field(description="Unique dataset identifier, matches directory name")
    path: str = Field(description="Path to dataset directory (relative to project root)")
    description: Optional[str] = Field(default=None, description="Content of ABOUT.md if present")
    n_samples: Optional[int] = None
    n_dims: Optional[int] = None
    n_categorical: Optional[int] = None


class DatasetUpdateSchema(BaseModel):
    description: Optional[str] = None
    n_samples: Optional[int] = None
    n_dims: Optional[int] = None
    n_categorical: Optional[int] = None


class SomRunCreateSchema(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {"id": "20260530_125716", "dataset_name": "WineQuality",
                    "map_m": 15, "map_n": 15, "mqe": 0.1247,
                    "topographic_error": 0.023, "dead_neuron_ratio": 0.09,
                    "duration_s": 752.8, "run_path": "data/datasets/WineQuality/results/20260530_125716"}
    })

    id: str = Field(description="Timestamp ID, e.g. '20260530_125716'")
    dataset_name: str = Field(description="Must match an existing dataset name")
    map_m: int
    map_n: int
    mqe: Optional[float] = None
    topographic_error: Optional[float] = None
    dead_neuron_ratio: Optional[float] = None
    duration_s: Optional[float] = None
    run_path: str
    ea_uid: Optional[str] = Field(default=None, description="Set when this run is an analyzed EA individual")


class ClusterBulkItem(BaseModel):
    neuron_key: str
    sample_ids: list[int]


class SampleAssignmentBulkItem(BaseModel):
    sample_id: int
    bmu_i: int
    bmu_j: int
    qe: Optional[float] = None
    qe_dims: Optional[dict[str, Optional[float]]] = Field(
        default=None, description="Per-dimension QE, e.g. {'alcohol': 0.04, 'pH': 0.01}")
    is_outlier: bool = False


class AnomalyBulkItem(BaseModel):
    sample_id: int
    qe: Optional[float] = None
    reason: Optional[list[Any]] = Field(
        default=None, description="List of reason dicts, e.g. [{'dim': 'x', 'type': 'global_max'}]")


class EaRunCreateSchema(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {"id": "20260515_111812", "dataset_name": "LungCancerDataset",
                    "run_path": "data/datasets/LungCancerDataset/results/20260515_111812"}
    })

    id: str
    dataset_name: str
    run_path: str


class EaSeedCreateSchema(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {"seed_value": 42, "n_generations": 50, "final_hv": 0.65, "pareto_size": 5}
    })

    seed_value: Optional[int] = None
    n_generations: Optional[int] = None
    final_hv: Optional[float] = None
    pareto_size: Optional[int] = None


class EaIndividualBulkItem(BaseModel):
    uid: str
    generation: int
    map_m: int
    map_n: int
    mqe: Optional[float] = None
    mqe_ratio: Optional[float] = None
    topographic_error: Optional[float] = None
    dead_ratio: Optional[float] = None
    topo_corr: Optional[float] = None
    constraint_violation: Optional[float] = None
    is_penalized: bool = False
    is_pareto_final: bool = False
    hyperparams: Optional[dict] = None
    duration_s: Optional[float] = None


class EaParetoMetricsBulkItem(BaseModel):
    generation: int
    front_size: Optional[int] = None
    hv: Optional[float] = None
    spacing: Optional[float] = None
    spread_mqe: Optional[float] = None
    spread_te: Optional[float] = None


class ImportJobSchema(BaseModel):
    job_id: str
    status: str = Field(description="One of: queued | running | done | error")
    message: Optional[str] = None


class HealthSchema(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {"status": "ok", "db_tables": 11, "version": "2.0"}
    })

    status: str
    db_tables: int = Field(description="Number of tables in DB — 11 expected when fully initialized")
    version: str = "2.0"
