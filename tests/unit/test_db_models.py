"""Unit tests for ORM models — no HTTP, direct DB session."""
import json
import pytest
from sqlalchemy.exc import IntegrityError

from app.api.models import Dataset, SomRun, SampleAssignment, Cluster, EaRun, EaSeed, EaIndividual


class TestDatasetModel:
    def test_insert_basic(self, db):
        ds = Dataset(name="Wine", path="/data/Wine", n_samples=1599, n_dims=11)
        db.add(ds)
        db.commit()
        assert db.query(Dataset).filter_by(name="Wine").first() is not None

    def test_unique_name(self, db):
        db.add(Dataset(name="Dup", path="/a"))
        db.commit()
        db.add(Dataset(name="Dup", path="/b"))
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

    def test_name_not_null(self, db):
        db.add(Dataset(name=None, path="/x"))
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()


class TestSomRunModel:
    def test_requires_dataset_fk(self, db):
        """Insert SOM run with non-existent dataset_id → IntegrityError."""
        run = SomRun(id="r1", dataset_id=9999, map_m=5, map_n=5, run_path="/x")
        db.add(run)
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

    def test_boundary_neuron_keys(self, db):
        """Neuron keys 0_0, 14_14, 0_14 store and retrieve without error."""
        ds = Dataset(name="DS", path="/x")
        db.add(ds)
        db.flush()
        run = SomRun(id="r2", dataset_id=ds.id, map_m=15, map_n=15, run_path="/x")
        db.add(run)
        db.flush()
        for key in ["0_0", "14_14", "0_14"]:
            db.add(SampleAssignment(
                run_id="r2", sample_id=1, bmu_i=0, bmu_j=0, bmu_key=key, qe=0.1,
            ))
        db.commit()
        keys = {r.bmu_key for r in db.query(SampleAssignment).filter_by(run_id="r2")}
        assert keys == {"0_0", "14_14", "0_14"}


class TestSampleAssignmentModel:
    def test_qe_dims_json_roundtrip(self, db):
        ds = Dataset(name="D", path="/x")
        db.add(ds)
        db.flush()
        run = SomRun(id="r", dataset_id=ds.id, map_m=3, map_n=3, run_path="/x")
        db.add(run)
        db.flush()

        original = {"alcohol": 0.12, "pH": None, "density": 0.05}
        db.add(SampleAssignment(run_id="r", sample_id=1, bmu_i=0, bmu_j=0,
                                bmu_key="0_0", qe=0.1, qe_dims=json.dumps(original)))
        db.commit()

        row = db.query(SampleAssignment).filter_by(run_id="r").first()
        loaded = json.loads(row.qe_dims)
        assert loaded == original

    def test_nan_qe_stored_as_null(self, db):
        ds = Dataset(name="D2", path="/x")
        db.add(ds)
        db.flush()
        run = SomRun(id="r2", dataset_id=ds.id, map_m=3, map_n=3, run_path="/x")
        db.add(run)
        db.flush()
        db.add(SampleAssignment(run_id="r2", sample_id=1, bmu_i=0, bmu_j=0, bmu_key="0_0", qe=None))
        db.commit()
        row = db.query(SampleAssignment).filter_by(run_id="r2").first()
        assert row.qe is None


class TestClusterModel:
    def test_sample_ids_json_array(self, db):
        ds = Dataset(name="D3", path="/x")
        db.add(ds)
        db.flush()
        run = SomRun(id="r3", dataset_id=ds.id, map_m=3, map_n=3, run_path="/x")
        db.add(run)
        db.flush()
        db.add(Cluster(run_id="r3", neuron_key="1_1", sample_ids=json.dumps([10, 20, 30]), sample_count=3))
        db.commit()
        row = db.query(Cluster).filter_by(run_id="r3", neuron_key="1_1").first()
        assert json.loads(row.sample_ids) == [10, 20, 30]


class TestEaIndividualModel:
    def test_composite_pk_different_seeds(self, db):
        """Same uid for two different seeds — allowed (composite PK)."""
        ds = Dataset(name="D4", path="/x")
        db.add(ds)
        db.flush()
        ea_run = EaRun(id="ea1", dataset_id=ds.id, run_path="/x")
        db.add(ea_run)
        db.flush()
        s1 = EaSeed(ea_run_id="ea1", seed_value=1)
        s2 = EaSeed(ea_run_id="ea1", seed_value=2)
        db.add_all([s1, s2])
        db.flush()

        db.add(EaIndividual(uid="same_uid", seed_id=s1.id, generation=1, map_m=5, map_n=5))
        db.add(EaIndividual(uid="same_uid", seed_id=s2.id, generation=1, map_m=5, map_n=5))
        db.commit()
        count = db.query(EaIndividual).filter_by(uid="same_uid").count()
        assert count == 2

    def test_duplicate_uid_same_seed_raises(self, db):
        """Same uid + same seed → IntegrityError."""
        ds = Dataset(name="D5", path="/x")
        db.add(ds)
        db.flush()
        ea_run = EaRun(id="ea2", dataset_id=ds.id, run_path="/x")
        db.add(ea_run)
        db.flush()
        seed = EaSeed(ea_run_id="ea2", seed_value=1)
        db.add(seed)
        db.flush()

        db.add(EaIndividual(uid="dup", seed_id=seed.id, generation=1, map_m=5, map_n=5))
        db.commit()
        db.add(EaIndividual(uid="dup", seed_id=seed.id, generation=2, map_m=5, map_n=5))
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()
