"""Edge case tests — extreme inputs and degenerate data states."""
import json
import pytest
from sqlalchemy.orm import Session

from app.api.models import Dataset, SomRun, SampleAssignment, Cluster, NeuronQe, Anomaly


def _insert_run(db: Session, run_id: str, m: int, n: int, *, dead_ratio: float = 0.0) -> str:
    ds = db.query(Dataset).filter_by(name="EdgeDS").first()
    if not ds:
        ds = Dataset(name="EdgeDS", path="/edge", n_samples=1, n_dims=2)
        db.add(ds)
        db.flush()
    run = SomRun(id=run_id, dataset_id=ds.id, map_m=m, map_n=n,
                 dead_neuron_ratio=dead_ratio, run_path="/edge/results/" + run_id)
    db.add(run)
    db.commit()
    return run_id


class TestAllDeadNeurons:
    def test_summary_all_dead(self, client, db):
        _insert_run(db, "dead_run", 2, 2)
        # No clusters → all 4 neurons are dead
        r = client.get("/runs/dead_run/summary")
        assert r.status_code == 200
        data = r.json()
        assert data["n_clusters_active"] == 0
        assert data["n_dead_neurons"] == 4

    def test_clusters_all_dead_returns_empty_list(self, client, db):
        _insert_run(db, "dead_run2", 2, 2)
        r = client.get("/runs/dead_run2/clusters")
        assert r.status_code == 200
        assert r.json() == []

    def test_anomalies_empty_run(self, client, db):
        _insert_run(db, "no_anomalies", 3, 3)
        r = client.get("/runs/no_anomalies/anomalies")
        assert r.status_code == 200
        assert r.json() == []


class TestSingleNeuronMap:
    def test_single_neuron_all_samples(self, client, db):
        """1×1 map: all samples must map to neuron '0_0'."""
        _insert_run(db, "single_neuron", 1, 1)
        for sid in range(1, 6):
            db.add(SampleAssignment(run_id="single_neuron", sample_id=sid,
                                    bmu_i=0, bmu_j=0, bmu_key="0_0", qe=0.1))
        db.add(Cluster(run_id="single_neuron", neuron_key="0_0",
                       sample_ids=json.dumps(list(range(1, 6))), sample_count=5))
        db.commit()

        r = client.get("/runs/single_neuron/clusters")
        assert r.status_code == 200
        clusters = r.json()
        assert len(clusters) == 1
        assert clusters[0]["neuron_key"] == "0_0"
        assert clusters[0]["sample_count"] == 5

    def test_single_neuron_detail(self, client, db):
        _insert_run(db, "single_n2", 1, 1)
        db.add(Cluster(run_id="single_n2", neuron_key="0_0",
                       sample_ids=json.dumps([1, 2, 3]), sample_count=3))
        db.commit()

        r = client.get("/runs/single_n2/clusters/0_0")
        assert r.status_code == 200
        assert r.json()["sample_ids"] == [1, 2, 3]


class TestRunWithoutAnalysis:
    def test_clusters_missing_analysis_returns_empty(self, client, db):
        """SOM run without any clusters in DB → empty list, not 500."""
        _insert_run(db, "no_analysis", 3, 3)
        r = client.get("/runs/no_analysis/clusters")
        assert r.status_code == 200
        assert r.json() == []

    def test_dimensions_no_sample_assignments(self, client, db):
        """No sample_assignments → empty dimensions list, not 500."""
        _insert_run(db, "no_sa", 3, 3)
        r = client.get("/runs/no_sa/dimensions")
        assert r.status_code == 200
        assert r.json() == []


class TestNanQeValues:
    def test_null_qe_in_assignments(self, client, db):
        """sample_assignments with NULL qe — cluster detail must not crash."""
        _insert_run(db, "null_qe", 2, 2)
        db.add(SampleAssignment(run_id="null_qe", sample_id=1,
                                bmu_i=0, bmu_j=0, bmu_key="0_0", qe=None,
                                qe_dims=json.dumps({"x": None, "y": 0.1})))
        db.add(Cluster(run_id="null_qe", neuron_key="0_0",
                       sample_ids=json.dumps([1]), sample_count=1))
        db.commit()

        r = client.get("/runs/null_qe/clusters/0_0")
        assert r.status_code == 200

    def test_null_qe_dims_skipped_in_aggregation(self, client, db):
        """Dimensions with None values must be skipped, not cause ZeroDivisionError."""
        _insert_run(db, "null_dims", 2, 2)
        db.add(SampleAssignment(run_id="null_dims", sample_id=1,
                                bmu_i=0, bmu_j=0, bmu_key="0_0", qe=0.1,
                                qe_dims=json.dumps({"x": None, "y": None})))
        db.commit()

        r = client.get("/runs/null_dims/dimensions")
        assert r.status_code == 200
        # All values null → no aggregatable data → empty list
        assert r.json() == []


class TestDatasetNameEdgeCases:
    def test_dataset_special_chars_in_name(self, client, db):
        """Dataset name with hyphens and underscores."""
        ds = Dataset(name="Swiss-Roll_v2", path="/x", n_samples=2000, n_dims=3)
        db.add(ds)
        db.commit()

        r = client.get("/datasets/Swiss-Roll_v2")
        assert r.status_code == 200
        assert r.json()["name"] == "Swiss-Roll_v2"

    def test_dataset_no_description(self, client, db):
        """Dataset without ABOUT.md — description should be null, no error."""
        ds = Dataset(name="NoDesc", path="/x", n_samples=100, n_dims=5)
        db.add(ds)
        db.flush()
        run = SomRun(id="no_desc_run", dataset_id=ds.id, map_m=5, map_n=5, run_path="/x")
        db.add(run)
        db.commit()

        r = client.get("/runs/no_desc_run/summary")
        assert r.status_code == 200
        assert r.json()["description"] is None


class TestInvalidQueryParams:
    def test_top_zero_returns_all(self, client, db):
        _insert_run(db, "tq_run", 3, 3)
        for key in ["0_0", "1_1"]:
            db.add(Cluster(run_id="tq_run", neuron_key=key,
                           sample_ids=json.dumps([1]), sample_count=1))
        db.commit()
        r = client.get("/runs/tq_run/clusters?top=0")
        assert r.status_code == 200
        assert len(r.json()) == 2

    def test_negative_top_rejected(self, client, db):
        _insert_run(db, "neg_top", 3, 3)
        r = client.get("/runs/neg_top/clusters?top=-1")
        assert r.status_code == 422
