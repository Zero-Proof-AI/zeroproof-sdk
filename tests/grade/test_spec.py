"""Spec: versioned model spec object, loading, and run stamping."""

from __future__ import annotations

import json

from zeroproof.simulations.score.spec import Spec, load_spec, spec_version, stamp_spec

CONSTITUTION = {
    "source": {"repo": "openai/model-spec", "file": "spec.md"},
    "traits": [
        {"id": "be_warm", "name": "Be warm", "principle": "Be warm.", "authority": "should"},
        {
            "id": "no_syco",
            "name": "Avoid sycophancy",
            "principle": "Do not flatter.",
            "authority": "must",
        },
    ],
}


def test_load_constitution_shape():
    s = load_spec(CONSTITUTION)
    assert s.behaviors() == ["be_warm", "no_syco"]
    assert s.principle("no_syco") == "Do not flatter."
    assert s.source["repo"] == "openai/model-spec"
    assert len(s.version) == 12


def test_version_is_content_derived_and_changes_on_edit():
    a = load_spec(CONSTITUTION)
    edited = json.loads(json.dumps(CONSTITUTION))
    edited["traits"][0]["principle"] = "Be extremely warm."
    b = load_spec(edited)
    assert a.version != b.version
    # same content -> same version, across calls
    assert load_spec(CONSTITUTION).version == a.version


def test_load_list_and_bare_strings():
    s = load_spec(["always cite sources", "never invent ids"])
    assert len(s.traits) == 2
    assert s.traits[0].principle == "always cite sources"
    assert s.behaviors() == ["trait_0", "trait_1"]


def test_load_from_path(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(CONSTITUTION), encoding="utf-8")
    s = load_spec(p)
    assert s.behaviors() == ["be_warm", "no_syco"]


def test_explicit_version_kept():
    data = {**CONSTITUTION, "version": "v1.2.3"}
    assert load_spec(data).version == "v1.2.3"


def test_spec_version_helper():
    assert spec_version(CONSTITUTION) == load_spec(CONSTITUTION).version


def test_stamp_spec():
    s = load_spec(CONSTITUTION)
    rows = stamp_spec([{"prompt": "a"}, {"prompt": "b"}], s)
    assert all(r["spec_id"] == s.id and r["spec_version"] == s.version for r in rows)


def test_to_dict_roundtrips_behaviors():
    s = load_spec(CONSTITUTION)
    d = s.to_dict()
    assert [t["id"] for t in d["traits"]] == s.behaviors()
    assert d["version"] == s.version


def test_behaviors_feed_delta_must_not_regress():
    # The trait ids are exactly what delta_report expects as marker names.
    s = load_spec(CONSTITUTION)
    assert isinstance(Spec("x").behaviors(), list)
    assert s.behaviors() == ["be_warm", "no_syco"]
