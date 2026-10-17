"""Settings of the maintenance process, and the rule that keeps the claim protocol safe."""

import os

import pytest
from pydantic import ValidationError

from services.embed.maintenance import MaintenanceSettings


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # a variable set in the developer's shell must not change what these tests see
    for name in [name for name in os.environ if name.startswith("MAINTENANCE_")]:
        monkeypatch.delenv(name)


def test_the_defaults_give_a_stale_point_an_hour_and_a_claim_ten_minutes() -> None:
    settings = MaintenanceSettings()
    assert settings.stale_grace_s == 3600.0
    assert settings.claim_timeout_s == 600.0
    assert (settings.compactor_batch_size, settings.compactor_interval_s) == (100, 30.0)
    assert settings.qdrant_collection == "chunks_esci"  # the same collection the worker writes


def test_the_process_has_its_own_health_port() -> None:
    # the api uses 8000, the worker 8081, the relay 8083
    assert MaintenanceSettings().health_port == 8082


def test_environment_variables_override_the_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAINTENANCE_STALE_GRACE_S", "120")
    monkeypatch.setenv("MAINTENANCE_QDRANT_COLLECTION", "chunks_blue")
    settings = MaintenanceSettings()
    assert (settings.stale_grace_s, settings.qdrant_collection) == (120.0, "chunks_blue")


@pytest.mark.parametrize(
    "name",
    [
        "MAINTENANCE_STALE_GRACE_S",
        "MAINTENANCE_CLAIM_TIMEOUT_S",
        "MAINTENANCE_COMPACTOR_BATCH_SIZE",
        "MAINTENANCE_COMPACTOR_INTERVAL_S",
    ],
)
def test_the_numbers_must_be_positive(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setenv(name, "0")
    with pytest.raises(ValidationError):
        MaintenanceSettings()


def test_the_claim_timeout_must_be_much_longer_than_the_qdrant_timeout() -> None:
    # a compactor that is only slow, not dead, could otherwise still send its delete after
    # another compactor took its claim over and an ingester staged the chunk again
    with pytest.raises(ValidationError, match="10 times qdrant_timeout_s"):
        MaintenanceSettings(qdrant_timeout_s=10, claim_timeout_s=99)
    assert MaintenanceSettings(qdrant_timeout_s=10, claim_timeout_s=100).claim_timeout_s == 100
