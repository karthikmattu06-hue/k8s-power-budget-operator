"""The kopf module imports and registers its handlers (no cluster needed)."""
# Uses kopf registry internals (_changing/_spawning); if a kopf upgrade breaks this test, update it.

import kopf


def test_handlers_register() -> None:
    from pbo.operator import main

    registry = kopf.get_default_registry()
    ids = {h.id for h in registry._changing.get_all_handlers()} | {h.id for h in registry._spawning.get_all_handlers()}
    assert any("created" in i for i in ids)
    assert any("tick" in i for i in ids)
    assert any("deleted" in i for i in ids)
    assert main.TICK_S == 5.0
