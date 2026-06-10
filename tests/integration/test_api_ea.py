"""Integration tests for EA endpoints."""
from tests.conftest import EA_RUN_ID


class TestEaList:
    def test_list_empty_initially(self, client):
        r = client.get("/ea")
        assert r.status_code == 200
        assert r.json() == []

    def test_list_after_insert(self, client, minimal_ea_run):
        r = client.get("/ea")
        assert r.status_code == 200
        ids = [e["id"] for e in r.json()]
        assert EA_RUN_ID in ids


class TestEaOverview:
    def test_overview_structure(self, client, minimal_ea_run):
        r = client.get(f"/ea/{EA_RUN_ID}")
        assert r.status_code == 200
        data = r.json()
        assert data["n_seeds"] == 1
        assert "seeds" in data

    def test_overview_not_found(self, client):
        r = client.get("/ea/ghost_ea")
        assert r.status_code == 404


class TestParetoEvolution:
    def test_pareto_evolution_structure(self, client, minimal_ea_run):
        r = client.get(f"/ea/{EA_RUN_ID}")
        seed_id = r.json()["seeds"][0]["id"]

        r = client.get(f"/ea/{EA_RUN_ID}/seeds/{seed_id}/pareto")
        assert r.status_code == 200
        data = r.json()
        assert "evolution" in data
        assert "pareto_final" in data
        assert len(data["evolution"]) == 3

    def test_pareto_hv_monotonic(self, client, minimal_ea_run):
        """HV must be non-decreasing across generations (NSGA-II guarantee)."""
        r = client.get(f"/ea/{EA_RUN_ID}")
        seed_id = r.json()["seeds"][0]["id"]

        r = client.get(f"/ea/{EA_RUN_ID}/seeds/{seed_id}/pareto")
        hvs = [g["hv"] for g in r.json()["evolution"]]
        assert hvs == sorted(hvs), f"HV not monotonic: {hvs}"

    def test_pareto_final_count(self, client, minimal_ea_run):
        r = client.get(f"/ea/{EA_RUN_ID}")
        seed_id = r.json()["seeds"][0]["id"]

        r = client.get(f"/ea/{EA_RUN_ID}/seeds/{seed_id}/pareto")
        assert len(r.json()["pareto_final"]) == 2

    def test_pareto_seed_not_found(self, client, minimal_ea_run):
        r = client.get(f"/ea/{EA_RUN_ID}/seeds/9999/pareto")
        assert r.status_code == 404

    def test_pareto_ea_run_not_found(self, client):
        r = client.get("/ea/ghost/seeds/1/pareto")
        assert r.status_code == 404


class TestEaIndividual:
    def test_get_individual_existing(self, client, minimal_ea_run):
        r = client.get(f"/ea/{EA_RUN_ID}/individuals/uid_b2")
        assert r.status_code == 200
        data = r.json()
        assert data["uid"] == "uid_b2"
        assert data["is_pareto_final"] == 1
        assert isinstance(data["hyperparams"], dict)

    def test_get_individual_not_found(self, client, minimal_ea_run):
        r = client.get(f"/ea/{EA_RUN_ID}/individuals/ghost_uid")
        assert r.status_code == 404

    def test_get_individual_ea_not_found(self, client):
        r = client.get("/ea/ghost_ea/individuals/uid_b2")
        assert r.status_code == 404
