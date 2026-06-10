"""Unit tests for Pydantic schema validation."""
import json
import pytest
from pydantic import ValidationError

from app.api.schemas import (
    ClusterDetailSchema, AnomalySchema, EaIndividualSchema, RunSummarySchema,
)


class TestClusterDetailSchema:
    def test_parse_sample_ids_from_json_string(self):
        schema = ClusterDetailSchema(neuron_key="1_1", sample_count=3, sample_ids=json.dumps([10, 20, 30]))
        assert schema.sample_ids == [10, 20, 30]

    def test_sample_ids_empty_list(self):
        schema = ClusterDetailSchema(neuron_key="0_0", sample_count=0, sample_ids=json.dumps([]))
        assert schema.sample_ids == []

    def test_sample_ids_as_list_passthrough(self):
        schema = ClusterDetailSchema(neuron_key="0_0", sample_count=2, sample_ids=[1, 2])
        assert schema.sample_ids == [1, 2]

    def test_sample_ids_none_defaults_empty(self):
        schema = ClusterDetailSchema(neuron_key="0_0", sample_count=0)
        assert schema.sample_ids == []


class TestAnomalySchema:
    def test_parse_reason_from_json_string(self):
        reason = json.dumps([{"dim": "x", "type": "global_max"}])
        schema = AnomalySchema(sample_id=5, qe=0.45, reason=reason)
        assert isinstance(schema.reason, list)
        assert schema.reason[0]["dim"] == "x"

    def test_reason_as_dict_passthrough(self):
        reason = [{"dim": "y", "type": "global_min"}]
        schema = AnomalySchema(sample_id=1, qe=0.1, reason=reason)
        assert schema.reason == reason

    def test_reason_none(self):
        schema = AnomalySchema(sample_id=1, qe=0.1, reason=None)
        assert schema.reason is None


class TestEaIndividualSchema:
    def test_parse_hyperparams_from_json_string(self):
        hp = json.dumps({"lr": 0.5, "radius": 3, "decay": "exp"})
        schema = EaIndividualSchema(uid="abc", hyperparams=hp)
        assert schema.hyperparams["lr"] == 0.5

    def test_hyperparams_none(self):
        schema = EaIndividualSchema(uid="abc")
        assert schema.hyperparams is None


class TestRunSummarySchema:
    def test_valid_summary(self):
        s = RunSummarySchema(
            run_id="test_001", dataset="Wine", map_size=[15, 15],
            mqe=0.12, topographic_error=0.02,
            n_samples=1599, n_dims=11,
            n_clusters_active=180, n_dead_neurons=45, n_anomalies=12,
        )
        assert s.map_size == [15, 15]
        assert s.n_dead_neurons == 45

    def test_optional_fields_default_none(self):
        s = RunSummarySchema(
            run_id="r", dataset="D", map_size=[3, 3],
            n_clusters_active=5, n_dead_neurons=4, n_anomalies=0,
        )
        assert s.description is None
        assert s.mqe is None
