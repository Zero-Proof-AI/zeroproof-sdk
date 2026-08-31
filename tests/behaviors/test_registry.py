from zeroproof_simulations.behaviors import packs


def test_packs_discover_without_shared_edits():
    found = packs()
    for name, module in found.items():
        assert callable(module.transform)
        assert callable(module.marker)
        assert module.SPEC["description"]
