from evonn_compare.contracts import models as compare_models
from evonn_shared import contracts as shared_contracts


def test_compare_contract_models_remain_shared_reexports() -> None:
    assert compare_models.ArtifactPaths is shared_contracts.ArtifactPaths
    assert compare_models.BenchmarkEntry is shared_contracts.BenchmarkEntry
    assert compare_models.BudgetEnvelope is shared_contracts.BudgetEnvelope
    assert compare_models.DeviceInfo is shared_contracts.DeviceInfo
    assert compare_models.FairnessEnvelope is shared_contracts.FairnessEnvelope
    assert compare_models.ResultRecord is shared_contracts.ResultRecord
    assert compare_models.RunManifest is shared_contracts.RunManifest
    assert compare_models.SearchTelemetry is shared_contracts.SearchTelemetry
