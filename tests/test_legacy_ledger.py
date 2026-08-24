from pathlib import Path

import pytest


LEDGER = """# Example
| id | one-line | mechanism family | target | status / venue | source |
|---|---|---|---|---|---|
| **SyncMem** (D04-CF-10) | parallel readers | `PARALLEL_MEMORY_SYNC` | long-document reading / RAG | **SURVIVES_WITH_RESIDUAL** | `sync.md` |
| **Contrast Witness** (D06-CWKV-01) | pairwise witness | `PAIRWISE_DISTINGUISHABILITY` | KV-cache compression | SURVIVES | `cw.md` |

## Killed / externally-covered
| candidate | mechanism + target | covered by (verified) |
|---|---|---|
| XD-01 causal-use verifier | counterfactual evidence reward+metric / RAG faithfulness | Correctness; EviOmni |
"""


def test_parse_recognized_live_and_killed_rows_with_line_provenance() -> None:
    from idea_factory.legacy_ledger import parse_legacy_markdown

    parsed = parse_legacy_markdown(LEDGER.encode("utf-8"), source_path="C:/ledger.md")
    assert [row["legacy_id"] for row in parsed.records] == ["D04-CF-10", "D06-CWKV-01", "XD-01"]
    sync = parsed.records[0]
    assert sync["display_name"] == "SyncMem"
    assert sync["mechanism_family"] == "PARALLEL_MEMORY_SYNC"
    assert sync["target"] == "long-document reading / RAG"
    assert sync["source_line"] == 4
    killed = parsed.records[-1]
    assert killed["status"] == "EXTERNALLY_COVERED"
    assert killed["mechanism_family"] == "counterfactual evidence reward+metric"
    assert killed["target"] == "RAG faithfulness"


def test_malformed_and_unrecognized_rows_are_preserved_not_guessed() -> None:
    from idea_factory.legacy_ledger import parse_legacy_markdown

    source = LEDGER + "| XD-01 duplicate | x / y | z |\n"
    parsed_duplicate_prefix = parse_legacy_markdown(source.encode("utf-8"), source_path="C:/ledger.md")
    xd_ids = sorted(row["legacy_id"] for row in parsed_duplicate_prefix.records if row["display_name"].startswith("XD-01"))
    assert len(xd_ids) == 2 and len(set(xd_ids)) == 2
    assert all(value.startswith("XD-01-") for value in xd_ids)
    parsed = parse_legacy_markdown((LEDGER + "\nnot a recognized table\n| x | y | z |\n").encode("utf-8"), source_path="C:/ledger.md")
    assert any(item["line"] == "| x | y | z |" for item in parsed.unparsed)


def test_recognized_table_rejects_wrong_column_count_and_allows_escaped_pipes() -> None:
    from idea_factory.legacy_ledger import parse_legacy_markdown

    malformed_line = "| too | few | columns |"
    malformed = parse_legacy_markdown((LEDGER.split("## Killed")[0] + malformed_line + "\n").encode("utf-8"), source_path="C:/ledger.md")
    assert malformed.errors == ({"line_number": 7, "error": "live ledger row has wrong column count", "line": malformed_line},)
    assert malformed.unparsed[-1] == {"line_number": 7, "line": malformed_line}
    source = LEDGER.replace("parallel readers", r"parallel readers \| synchronizers")
    parsed = parse_legacy_markdown(source.encode("utf-8"), source_path="C:/ledger.md")
    assert parsed.records[0]["one_line"] == "parallel readers | synchronizers"


def test_killed_short_prefix_disambiguation_is_generic_and_order_independent() -> None:
    from idea_factory.legacy_ledger import parse_legacy_markdown

    header = "| candidate | mechanism + target | covered by (verified) |\n|---|---|---|\n"
    rows = [
        "| D03 CL01-1 binding | position binding / VLM | p1 |",
        "| D03 CL10-1 segment | segment mask / VLM | p2 |",
        "| ABC-9 unique candidate | mechanism / target | p3 |",
        "| (seed) first candidate | mechanism / target | p4 |",
        "| (seed) second candidate | mechanism / target | p5 |",
    ]
    first = parse_legacy_markdown((header + "\n".join(rows) + "\n").encode(), source_path="x")
    second = parse_legacy_markdown((header + "\n".join(reversed(rows)) + "\n").encode(), source_path="x")
    by_name_first = {row["display_name"]: row["legacy_id"] for row in first.records}
    by_name_second = {row["display_name"]: row["legacy_id"] for row in second.records}
    assert by_name_first == by_name_second
    assert by_name_first["ABC-9 unique candidate"] == "ABC-9"
    assert all(by_name_first[name].startswith("D03-") for name in by_name_first if name.startswith("D03"))
    assert "(seed) first candidate" in by_name_first
    assert all(by_name_first[name].startswith("(seed)-") for name in by_name_first if name.startswith("(seed)"))


def test_identical_killed_candidates_are_preserved_as_parse_errors() -> None:
    from idea_factory.legacy_ledger import parse_legacy_markdown

    row = "| XD-01 same | mechanism / target | evidence |"
    source = f"| candidate | mechanism + target | covered by (verified) |\n|---|---|---|\n{row}\n{row}\n"
    parsed = parse_legacy_markdown(source.encode(), source_path="x")
    assert not parsed.records
    assert [item["line"] for item in parsed.unparsed] == [row, row]
    assert [error["line_number"] for error in parsed.errors] == [3, 4]


def test_live_ids_are_globally_reserved_from_killed_prefixes_across_table_order() -> None:
    from idea_factory.legacy_ledger import parse_legacy_markdown

    live = "| id | one-line | mechanism family | target | status / venue | source |\n|---|---|---|---|---|---|\n| **Live XD** (XD-01) | live | LIVE_MECH | live target | live | source |\n"
    killed = "| candidate | mechanism + target | covered by (verified) |\n|---|---|---|\n| XD-01 killed candidate | killed mechanism / killed target | evidence |\n"
    first = parse_legacy_markdown((live + "\nsection\n" + killed).encode(), source_path="x")
    second = parse_legacy_markdown((killed + "\nsection\n" + live).encode(), source_path="x")
    by_name_first = {row["display_name"]: row["legacy_id"] for row in first.records}
    by_name_second = {row["display_name"]: row["legacy_id"] for row in second.records}
    assert by_name_first == by_name_second
    assert by_name_first["Live XD"] == "XD-01"
    assert by_name_first["XD-01 killed candidate"].startswith("XD-01-")
    assert len(set(by_name_first.values())) == len(by_name_first)


def test_final_derived_id_collision_becomes_auditable_residue(monkeypatch: pytest.MonkeyPatch) -> None:
    import idea_factory.legacy_ledger as parser

    monkeypatch.setattr(parser, "_disambiguated_killed_id", lambda prefix, candidate: prefix + "-collision")
    rows = [
        "| SAME first candidate | mechanism / target | evidence one |",
        "| SAME second candidate | mechanism / target | evidence two |",
    ]
    source = "| candidate | mechanism + target | covered by (verified) |\n|---|---|---|\n" + "\n".join(rows) + "\n"
    parsed = parser.parse_legacy_markdown(source.encode(), source_path="x")
    assert parsed.records == ()
    assert [item["line"] for item in parsed.unparsed] == rows
    assert {error["error"] for error in parsed.errors} == {"final global legacy ID collision"}


def test_import_is_config_bound_and_source_is_unchanged(tmp_path: Path) -> None:
    from idea_factory.corpus import CorpusRouterConfig
    from idea_factory.legacy_ledger import import_legacy_ledger, validate_legacy_import

    papers = tmp_path / "papers"
    (papers / "_ideas").mkdir(parents=True)
    source = papers / "_ideas" / "idea_ledger.md"
    source.write_text(LEDGER, encoding="utf-8")
    notes = tmp_path / "notes"; notes.mkdir()
    candidates = papers / "candidates.txt"; candidates.write_text("", encoding="utf-8")
    config = CorpusRouterConfig(notes, (candidates,), source, 1, 1, ("KV_CACHE",), ())
    run = tmp_path / "run"; run.mkdir()
    before = source.read_bytes()
    paths = import_legacy_ledger(run, config)
    assert paths["index"].is_file()
    assert source.read_bytes() == before
    bundle = validate_legacy_import(run, config)
    assert len(bundle.records) == 3
    assert bundle.manifest["parser_version"] == "idea_factory.legacy_ledger.v2"


def test_unparsed_artifact_is_verbatim_and_manifest_keeps_line_errors(tmp_path: Path) -> None:
    from idea_factory.corpus import CorpusRouterConfig
    from idea_factory.legacy_ledger import import_legacy_ledger, validate_legacy_import

    papers = tmp_path / "papers"; source = papers / "_ideas" / "idea_ledger.md"; source.parent.mkdir(parents=True)
    bad = "| malformed | live | row |"
    source.write_text(LEDGER.split("## Killed")[0] + bad + "\n", encoding="utf-8")
    notes = tmp_path / "notes"; notes.mkdir(); candidates = papers / "list.txt"; candidates.write_text("", encoding="utf-8")
    config = CorpusRouterConfig(notes, (candidates,), source, 1, 1, ("KV_CACHE",), ())
    run = tmp_path / "run"; run.mkdir()
    paths = import_legacy_ledger(run, config)
    parsed = validate_legacy_import(run, config)
    expected = "".join(row["line"] + "\n" for row in parsed.unparsed)
    assert paths["unparsed"].read_text(encoding="utf-8") == expected
    assert not paths["unparsed"].read_text(encoding="utf-8").splitlines()[-1].startswith("7:")
    assert parsed.manifest["parse_errors"][-1]["line"] == bad
