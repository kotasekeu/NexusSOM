"""Integration tests for write operations (POST, PUT, DELETE)."""
import json
import pytest
from tests.conftest import SOM_RUN_ID, EA_RUN_ID, DATASET_NAME


# ---------------------------------------------------------------------------
# Datasets — POST / PUT
# ---------------------------------------------------------------------------

class TestDatasetCreate:
    def test_create_dataset(self, client):
        r = client.post("/datasets", json={
            "name": "NewDS", "path": "data/datasets/NewDS",
            "n_samples": 500, "n_dims": 5,
        })
        assert r.status_code == 201
        data = r.json()
        assert data["name"] == "NewDS"
        assert data["n_samples"] == 500
        assert data["id"] is not None

    def test_create_dataset_conflict(self, client, minimal_som_run):
        r = client.post("/datasets", json={"name": DATASET_NAME, "path": "/x"})
        assert r.status_code == 409

    def test_create_dataset_missing_name(self, client):
        r = client.post("/datasets", json={"path": "/x"})
        assert r.status_code == 422

    def test_created_dataset_visible_in_list(self, client):
        client.post("/datasets", json={"name": "ListDS", "path": "/x"})
        r = client.get("/datasets")
        names = [d["name"] for d in r.json()]
        assert "ListDS" in names


class TestDatasetUpdate:
    def test_update_description(self, client, minimal_som_run):
        r = client.put(f"/datasets/{DATASET_NAME}", json={"description": "Updated description"})
        assert r.status_code == 200
        assert r.json()["description"] == "Updated description"

    def test_update_partial_fields(self, client, minimal_som_run):
        r = client.put(f"/datasets/{DATASET_NAME}", json={"n_samples": 9999})
        assert r.status_code == 200
        assert r.json()["n_samples"] == 9999

    def test_update_dataset_not_found(self, client):
        r = client.put("/datasets/ghost", json={"description": "x"})
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# SOM Runs — POST + bulk sub-endpoints + DELETE
# ---------------------------------------------------------------------------

class TestSomRunCreate:
    def test_create_run(self, client, minimal_som_run):
        r = client.post("/runs", json={
            "id": "new_run_001", "dataset_name": DATASET_NAME,
            "map_m": 10, "map_n": 10, "mqe": 0.20, "topographic_error": 0.04,
            "run_path": "/fake/results/new_run_001",
        })
        assert r.status_code == 201
        assert r.json()["id"] == "new_run_001"

    def test_create_run_dataset_not_found(self, client):
        r = client.post("/runs", json={
            "id": "x", "dataset_name": "ghost", "map_m": 5, "map_n": 5, "run_path": "/x",
        })
        assert r.status_code == 404

    def test_create_run_duplicate_id(self, client, minimal_som_run):
        r = client.post("/runs", json={
            "id": SOM_RUN_ID, "dataset_name": DATASET_NAME,
            "map_m": 5, "map_n": 5, "run_path": "/x",
        })
        assert r.status_code == 409

    def test_create_run_appears_in_dataset_runs(self, client, minimal_som_run):
        client.post("/runs", json={
            "id": "run_v2", "dataset_name": DATASET_NAME,
            "map_m": 8, "map_n": 8, "run_path": "/fake/results/run_v2",
        })
        r = client.get(f"/datasets/{DATASET_NAME}/runs")
        ids = [run["id"] for run in r.json()["som_runs"]]
        assert "run_v2" in ids


class TestBulkClusters:
    def test_bulk_add_clusters(self, client, minimal_som_run):
        run_id = "bulk_test"
        client.post("/runs", json={
            "id": run_id, "dataset_name": DATASET_NAME,
            "map_m": 3, "map_n": 3, "run_path": "/x",
        })
        r = client.post(f"/runs/{run_id}/clusters", json=[
            {"neuron_key": "0_0", "sample_ids": [1, 2, 3]},
            {"neuron_key": "1_1", "sample_ids": [4, 5]},
        ])
        assert r.status_code == 201
        assert r.json()["inserted"] == 2

        r = client.get(f"/runs/{run_id}/clusters")
        assert len(r.json()) == 2

    def test_bulk_clusters_upsert(self, client, minimal_som_run):
        """Posting same neuron_key again replaces it."""
        run_id = "upsert_test"
        client.post("/runs", json={"id": run_id, "dataset_name": DATASET_NAME, "map_m": 3, "map_n": 3, "run_path": "/x"})
        client.post(f"/runs/{run_id}/clusters", json=[{"neuron_key": "0_0", "sample_ids": [1]}])
        client.post(f"/runs/{run_id}/clusters", json=[{"neuron_key": "0_0", "sample_ids": [1, 2, 3]}])

        r = client.get(f"/runs/{run_id}/clusters/0_0")
        assert r.json()["sample_count"] == 3

    def test_bulk_clusters_run_not_found(self, client):
        r = client.post("/runs/ghost/clusters", json=[{"neuron_key": "0_0", "sample_ids": [1]}])
        assert r.status_code == 404


class TestBulkAssignments:
    def test_bulk_add_assignments(self, client, minimal_som_run):
        run_id = "assign_test"
        client.post("/runs", json={"id": run_id, "dataset_name": DATASET_NAME, "map_m": 3, "map_n": 3, "run_path": "/x"})
        r = client.post(f"/runs/{run_id}/assignments", json=[
            {"sample_id": 1, "bmu_i": 0, "bmu_j": 0, "qe": 0.10, "qe_dims": {"x": 0.08, "y": 0.05}},
            {"sample_id": 2, "bmu_i": 1, "bmu_j": 1, "qe": 0.15, "is_outlier": True},
        ])
        assert r.status_code == 201
        assert r.json()["inserted"] == 2

    def test_bulk_assignments_dimension_endpoint(self, client, minimal_som_run):
        """Dimensions endpoint must work after adding assignments with qe_dims."""
        run_id = "dim_test"
        client.post("/runs", json={"id": run_id, "dataset_name": DATASET_NAME, "map_m": 2, "map_n": 2, "run_path": "/x"})
        client.post(f"/runs/{run_id}/assignments", json=[
            {"sample_id": 1, "bmu_i": 0, "bmu_j": 0, "qe": 0.1, "qe_dims": {"alcohol": 0.04, "pH": 0.02}},
        ])
        r = client.get(f"/runs/{run_id}/dimensions")
        names = [d["name"] for d in r.json()]
        assert "alcohol" in names and "pH" in names


class TestBulkAnomalies:
    def test_bulk_add_anomalies(self, client, minimal_som_run):
        run_id = "anom_test"
        client.post("/runs", json={"id": run_id, "dataset_name": DATASET_NAME, "map_m": 3, "map_n": 3, "run_path": "/x"})
        r = client.post(f"/runs/{run_id}/anomalies", json=[
            {"sample_id": 42, "qe": 0.9, "reason": [{"dim": "x", "type": "global_max"}]},
            {"sample_id": 99, "qe": 0.7},
        ])
        assert r.status_code == 201
        assert r.json()["inserted"] == 2

        r = client.get(f"/runs/{run_id}/anomalies")
        assert len(r.json()) == 2


class TestSomRunDelete:
    def test_delete_run(self, client, minimal_som_run):
        r = client.delete(f"/runs/{SOM_RUN_ID}")
        assert r.status_code == 204

        r = client.get(f"/runs/{SOM_RUN_ID}/summary")
        assert r.status_code == 404

    def test_delete_run_not_found(self, client):
        r = client.delete("/runs/ghost")
        assert r.status_code == 404

    def test_delete_clears_related_data(self, client, minimal_som_run):
        """Deleting a run must also remove clusters, assignments, anomalies."""
        client.delete(f"/runs/{SOM_RUN_ID}")
        # Run-keyed endpoints return 404, not empty data for deleted runs
        assert client.get(f"/runs/{SOM_RUN_ID}/clusters").status_code == 404
        assert client.get(f"/runs/{SOM_RUN_ID}/anomalies").status_code == 404


# ---------------------------------------------------------------------------
# EA — POST + DELETE
# ---------------------------------------------------------------------------

class TestEaRunCreate:
    def test_create_ea_run(self, client, minimal_som_run):
        r = client.post("/ea", json={
            "id": "new_ea_001", "dataset_name": DATASET_NAME,
            "run_path": "/fake/results/new_ea_001",
        })
        assert r.status_code == 201
        assert r.json()["id"] == "new_ea_001"

    def test_create_ea_run_conflict(self, client, minimal_ea_run):
        r = client.post("/ea", json={
            "id": EA_RUN_ID, "dataset_name": DATASET_NAME, "run_path": "/x",
        })
        assert r.status_code == 409

    def test_create_ea_run_dataset_not_found(self, client):
        r = client.post("/ea", json={"id": "x", "dataset_name": "ghost", "run_path": "/x"})
        assert r.status_code == 404


class TestEaSeedCreate:
    def test_create_seed(self, client, minimal_ea_run):
        r = client.post(f"/ea/{EA_RUN_ID}/seeds", json={"seed_value": 99, "n_generations": 30})
        assert r.status_code == 201
        data = r.json()
        assert data["seed_value"] == 99
        assert data["id"] is not None

    def test_create_seed_ea_not_found(self, client):
        r = client.post("/ea/ghost/seeds", json={"seed_value": 1})
        assert r.status_code == 404


class TestEaIndividualsBulk:
    def test_bulk_add_individuals(self, client, minimal_ea_run):
        seed_id = client.get(f"/ea/{EA_RUN_ID}").json()["seeds"][0]["id"]

        r = client.post(f"/ea/{EA_RUN_ID}/seeds/{seed_id}/individuals", json=[
            {"uid": "new_uid_x", "generation": 4, "map_m": 10, "map_n": 10,
             "mqe": 0.5, "topographic_error": 0.03, "is_pareto_final": False,
             "hyperparams": {"lr": 0.4}},
        ])
        assert r.status_code == 201
        assert r.json()["inserted"] == 1

    def test_bulk_add_individuals_idempotent(self, client, minimal_ea_run):
        """Duplicate uid+seed combination is skipped silently."""
        seed_id = client.get(f"/ea/{EA_RUN_ID}").json()["seeds"][0]["id"]
        payload = [{"uid": "uid_a1", "generation": 1, "map_m": 10, "map_n": 10}]

        r = client.post(f"/ea/{EA_RUN_ID}/seeds/{seed_id}/individuals", json=payload)
        assert r.json()["skipped"] == 1
        assert r.json()["inserted"] == 0

    def test_bulk_add_pareto_metrics(self, client, minimal_ea_run):
        seed_id = client.get(f"/ea/{EA_RUN_ID}").json()["seeds"][0]["id"]

        r = client.post(f"/ea/{EA_RUN_ID}/seeds/{seed_id}/pareto_metrics", json=[
            {"generation": 10, "front_size": 3, "hv": 0.80},
        ])
        assert r.status_code == 201
        assert r.json()["inserted"] == 1


class TestEaRunDelete:
    def test_delete_ea_run(self, client, minimal_ea_run):
        r = client.delete(f"/ea/{EA_RUN_ID}")
        assert r.status_code == 204
        assert client.get(f"/ea/{EA_RUN_ID}").status_code == 404

    def test_delete_ea_not_found(self, client):
        r = client.delete("/ea/ghost")
        assert r.status_code == 404
