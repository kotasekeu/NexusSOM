"""Integration tests for SOM endpoints — use minimal_som_run fixture."""
import pytest
from tests.conftest import SOM_RUN_ID, DATASET_NAME


class TestDatasets:
    def test_list_empty_initially(self, client):
        r = client.get("/datasets")
        assert r.status_code == 200
        assert r.json() == []

    def test_list_after_insert(self, client, minimal_som_run):
        r = client.get("/datasets")
        assert r.status_code == 200
        names = [d["name"] for d in r.json()]
        assert DATASET_NAME in names

    def test_get_dataset_ok(self, client, minimal_som_run):
        r = client.get(f"/datasets/{DATASET_NAME}")
        assert r.status_code == 200
        assert r.json()["name"] == DATASET_NAME
        assert r.json()["n_samples"] == 10

    def test_get_dataset_not_found(self, client):
        r = client.get("/datasets/neexistuje")
        assert r.status_code == 404

    def test_list_runs_for_dataset(self, client, minimal_som_run):
        r = client.get(f"/datasets/{DATASET_NAME}/runs")
        assert r.status_code == 200
        data = r.json()
        assert len(data["som_runs"]) == 1
        assert data["som_runs"][0]["id"] == SOM_RUN_ID

    def test_list_runs_dataset_not_found(self, client):
        r = client.get("/datasets/ghost/runs")
        assert r.status_code == 404


class TestRunSummary:
    def test_summary_structure(self, client, minimal_som_run):
        r = client.get(f"/runs/{SOM_RUN_ID}/summary")
        assert r.status_code == 200
        data = r.json()
        assert data["run_id"] == SOM_RUN_ID
        assert data["map_size"] == [3, 3]
        assert data["n_clusters_active"] == 7
        assert data["n_dead_neurons"] == 2     # 3×3=9 neurons, 7 active → 2 dead
        assert data["n_anomalies"] == 2

    def test_summary_token_budget(self, client, minimal_som_run):
        r = client.get(f"/runs/{SOM_RUN_ID}/summary")
        response_text = r.text
        estimated_tokens = len(response_text) // 4
        assert estimated_tokens <= 800, f"Summary too large: {estimated_tokens} tokens"

    def test_summary_run_not_found(self, client):
        r = client.get("/runs/ghost_run/summary")
        assert r.status_code == 404

    def test_summary_dataset_description(self, client, minimal_som_run):
        r = client.get(f"/runs/{SOM_RUN_ID}/summary")
        assert r.json()["description"] == "Test dataset"


class TestClusters:
    def test_list_clusters_returns_all_active(self, client, minimal_som_run):
        r = client.get(f"/runs/{SOM_RUN_ID}/clusters")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 7
        assert all("neuron_key" in c and "sample_count" in c for c in data)

    def test_list_clusters_sorted_by_sample_count(self, client, minimal_som_run):
        r = client.get(f"/runs/{SOM_RUN_ID}/clusters?sort_by=sample_count")
        assert r.status_code == 200
        counts = [c["sample_count"] for c in r.json()]
        assert counts == sorted(counts, reverse=True)

    def test_list_clusters_top_n(self, client, minimal_som_run):
        r = client.get(f"/runs/{SOM_RUN_ID}/clusters?top=3")
        assert r.status_code == 200
        assert len(r.json()) == 3

    def test_list_clusters_invalid_sort_by(self, client, minimal_som_run):
        r = client.get(f"/runs/{SOM_RUN_ID}/clusters?sort_by=neexistujici")
        assert r.status_code == 422

    def test_list_clusters_run_not_found(self, client):
        r = client.get("/runs/ghost/clusters")
        assert r.status_code == 404

    def test_cluster_detail_correct_stats(self, client, minimal_som_run):
        r = client.get(f"/runs/{SOM_RUN_ID}/clusters/1_1")
        assert r.status_code == 200
        data = r.json()
        assert data["neuron_key"] == "1_1"
        assert data["sample_count"] == 2
        assert set(data["sample_ids"]) == {4, 5}

    def test_cluster_detail_nonexistent_key(self, client, minimal_som_run):
        r = client.get(f"/runs/{SOM_RUN_ID}/clusters/9_9")
        assert r.status_code == 404

    def test_cluster_detail_run_not_found(self, client):
        r = client.get("/runs/ghost/clusters/0_0")
        assert r.status_code == 404


class TestAnomalies:
    def test_list_anomalies(self, client, minimal_som_run):
        r = client.get(f"/runs/{SOM_RUN_ID}/anomalies")
        assert r.status_code == 200
        assert len(r.json()) == 2

    def test_anomalies_limit(self, client, minimal_som_run):
        r = client.get(f"/runs/{SOM_RUN_ID}/anomalies?limit=1")
        assert r.status_code == 200
        assert len(r.json()) == 1

    def test_anomalies_negative_limit(self, client, minimal_som_run):
        r = client.get(f"/runs/{SOM_RUN_ID}/anomalies?limit=-1")
        assert r.status_code == 422

    def test_anomalies_run_not_found(self, client):
        r = client.get("/runs/ghost/anomalies")
        assert r.status_code == 404

    def test_anomaly_reason_parsed(self, client, minimal_som_run):
        r = client.get(f"/runs/{SOM_RUN_ID}/anomalies")
        data = r.json()
        assert any(isinstance(a["reason"], list) for a in data)


class TestDimensions:
    def test_list_dimensions(self, client, minimal_som_run):
        r = client.get(f"/runs/{SOM_RUN_ID}/dimensions")
        assert r.status_code == 200
        data = r.json()
        names = [d["name"] for d in data]
        assert "x" in names
        assert "y" in names

    def test_get_dimension_existing(self, client, minimal_som_run):
        r = client.get(f"/runs/{SOM_RUN_ID}/dimensions/x")
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "x"
        assert data["qe_mean"] is not None

    def test_get_dimension_not_found(self, client, minimal_som_run):
        r = client.get(f"/runs/{SOM_RUN_ID}/dimensions/neexistuje")
        assert r.status_code == 404

    def test_dimensions_run_not_found(self, client):
        r = client.get("/runs/ghost/dimensions")
        assert r.status_code == 404


class TestRunCompare:
    def test_compare_two_runs(self, client, minimal_som_run):
        r = client.get(f"/runs/compare?ids={SOM_RUN_ID},{SOM_RUN_ID}")
        # Same run twice — valid structure, returns 2 items (deduplication is caller's problem)
        assert r.status_code == 200

    def test_compare_single_id_rejected(self, client, minimal_som_run):
        r = client.get(f"/runs/compare?ids={SOM_RUN_ID}")
        assert r.status_code == 422

    def test_compare_missing_run(self, client, minimal_som_run):
        r = client.get(f"/runs/compare?ids={SOM_RUN_ID},ghost_run")
        assert r.status_code == 404
