"""The docs page is a mirror of the root constitution, never a fork of it."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _docs_body() -> str:
    text = (ROOT / "docs" / "reference" / "constitution.md").read_text(encoding="utf-8")
    body = text.split("---", 2)[2]  # drop the front matter
    body = body.split("</Note>", 1)[1]  # drop the mirror note
    return body.strip()


def _root_body() -> str:
    text = (ROOT / "CONSTITUTION.md").read_text(encoding="utf-8")
    return text.split("\n", 1)[1].strip()  # drop the H1; Mintlify renders the title


def test_docs_constitution_mirrors_root() -> None:
    assert _docs_body() == _root_body(), (
        "docs/reference/constitution.md drifted from CONSTITUTION.md; copy the root file's body "
        "under the page's front matter and mirror note"
    )
