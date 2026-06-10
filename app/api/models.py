from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey, Index
from app.api.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    path = Column(String, nullable=False)
    description = Column(Text)
    n_samples = Column(Integer)
    n_dims = Column(Integer)
    n_categorical = Column(Integer)
    created_at = Column(String, default=lambda: datetime.utcnow().isoformat())


class SomRun(Base):
    __tablename__ = "som_runs"

    id = Column(String, primary_key=True)          # timestamp "20260530_125716"
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    ea_uid = Column(String)                         # NULL for standalone SOM
    map_m = Column(Integer, nullable=False)
    map_n = Column(Integer, nullable=False)
    mqe = Column(Float)
    topographic_error = Column(Float)
    dead_neuron_ratio = Column(Float)
    duration_s = Column(Float)
    run_path = Column(String, nullable=False)
    created_at = Column(String)


class SampleAssignment(Base):
    __tablename__ = "sample_assignments"

    id = Column(Integer, primary_key=True)
    run_id = Column(String, ForeignKey("som_runs.id"), nullable=False)
    sample_id = Column(Integer, nullable=False)
    bmu_i = Column(Integer, nullable=False)
    bmu_j = Column(Integer, nullable=False)
    bmu_key = Column(String, nullable=False)        # "i_j"
    qe = Column(Float)
    qe_dims = Column(Text)                          # JSON: {"x": 0.12, "y": 0.05}
    is_outlier = Column(Integer, default=0)


Index("idx_sa_run_bmu", SampleAssignment.run_id, SampleAssignment.bmu_key)
Index("idx_sa_run_sample", SampleAssignment.run_id, SampleAssignment.sample_id)


class NeuronQe(Base):
    __tablename__ = "neuron_qe"

    run_id = Column(String, ForeignKey("som_runs.id"), primary_key=True)
    neuron_key = Column(String, primary_key=True)
    qe_mean = Column(Float)
    qe_max = Column(Float)
    sample_count = Column(Integer)


class Cluster(Base):
    __tablename__ = "clusters"

    run_id = Column(String, ForeignKey("som_runs.id"), primary_key=True)
    neuron_key = Column(String, primary_key=True)
    sample_ids = Column(Text, nullable=False)       # JSON array of ints
    sample_count = Column(Integer)


class Anomaly(Base):
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True)
    run_id = Column(String, ForeignKey("som_runs.id"), nullable=False)
    sample_id = Column(Integer)
    reason = Column(Text)                           # JSON list of reason dicts
    qe = Column(Float)


class EaRun(Base):
    __tablename__ = "ea_runs"

    id = Column(String, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    run_path = Column(String, nullable=False)
    created_at = Column(String)


class EaSeed(Base):
    __tablename__ = "ea_seeds"

    id = Column(Integer, primary_key=True)
    ea_run_id = Column(String, ForeignKey("ea_runs.id"), nullable=False)
    seed_value = Column(Integer)
    n_generations = Column(Integer)
    final_hv = Column(Float)
    pareto_size = Column(Integer)


class EaIndividual(Base):
    __tablename__ = "ea_individuals"

    uid = Column(String, primary_key=True)
    seed_id = Column(Integer, ForeignKey("ea_seeds.id"), primary_key=True)
    generation = Column(Integer)
    map_m = Column(Integer)
    map_n = Column(Integer)
    mqe = Column(Float)
    mqe_ratio = Column(Float)
    topographic_error = Column(Float)
    dead_ratio = Column(Float)
    topo_corr = Column(Float)
    constraint_violation = Column(Float)
    is_penalized = Column(Integer, default=0)
    is_pareto_final = Column(Integer, default=0)
    hyperparams = Column(Text)                      # JSON
    duration_s = Column(Float)


Index("idx_ea_ind_seed_gen", EaIndividual.seed_id, EaIndividual.generation)


class EaParetoMetrics(Base):
    __tablename__ = "ea_pareto_metrics"

    seed_id = Column(Integer, ForeignKey("ea_seeds.id"), primary_key=True)
    generation = Column(Integer, primary_key=True)
    front_size = Column(Integer)
    hv = Column(Float)
    spacing = Column(Float)
    spread_mqe = Column(Float)
    spread_te = Column(Float)


class CalibrationProbe(Base):
    __tablename__ = "calibration_probes"

    ea_run_id = Column(String, ForeignKey("ea_runs.id"), primary_key=True)
    probe_idx = Column(Integer, primary_key=True)
    org_max = Column(Float)
