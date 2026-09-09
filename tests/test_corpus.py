from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

import idea_factory.artifacts as artifacts_module
import idea_factory.corpus as corpus_module
from idea_factory.artifacts import read_jsonl, stable_id
from idea_factory.corpus import (
    _choose_stratified,
    CorpusRouterConfig,
    RouterResult,
    enumerate_candidates,
    ingest_router_results,
    select_pilot,
    validate_corpus_selection_bundle,
    write_router_jobs,
)


REPO_ROOT = Path(__file__).parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
PROMPT = REPO_ROOT / "prompts" / "corpus_router.md"
PROMPT_SHA256 = hashlib.sha256(PROMPT.read_bytes()).hexdigest()
PROMPT_VERSION = "idea_factory.corpus_router_prompt.v2"
ROUTER_JOB_SCHEMA_VERSION = "idea_factory.corpus_router_job.v2"
ROUTER_RESULT_SCHEMA_VERSION = "idea_factory.corpus_router_result.v2"


def directory_link_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError:
        if os.name != "nt":
            pytest.skip("directory symlinks unavailable")
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip(f"directory links unavailable: {completed.stderr.strip()}")


def expected_job_id(slug: str, note_sha256: str, prompt_sha256: str) -> str:
    return stable_id(
        "router_job", slug, note_sha256, prompt_sha256, ROUTER_RESULT_SCHEMA_VERSION
    )


def config(**overrides: object) -> CorpusRouterConfig:
    values: dict[str, object] = {
        "notes_root": FIXTURES / "notes",
        "candidate_lists": (
            FIXTURES / "candidates" / "list-one.txt",
            FIXTURES / "candidates" / "list-two.txt",
        ),
        "legacy_ledger": FIXTURES / "legacy-ledger.md",
        "target_min": 1,
        "target_max": 4,
        "allowed_labels": ("KV_CACHE", "LONG_MEMORY", "BRIDGE"),
        "bridge_regression_slugs": ("paper-bridge",),
    }
    values.update(overrides)
    return CorpusRouterConfig(**values)


def scoped_config(tmp_path: Path, entries: list[Path]) -> CorpusRouterConfig:
    notes_root = tmp_path / "notes"
    corpus_root = tmp_path / "papers"
    notes_root.mkdir(exist_ok=True)
    (corpus_root / "_ideas").mkdir(parents=True, exist_ok=True)
    legacy_ledger = corpus_root / "_ideas" / "idea_ledger.md"
    legacy_ledger.write_text("legacy", encoding="utf-8")
    candidate_list = corpus_root / "round" / "note_paths.txt"
    candidate_list.parent.mkdir(parents=True, exist_ok=True)
    candidate_list.write_text(
        "".join(f"{entry}\n" for entry in entries), encoding="utf-8"
    )
    return CorpusRouterConfig(
        notes_root=notes_root,
        candidate_lists=(candidate_list,),
        legacy_ledger=legacy_ledger,
        target_min=1,
        target_max=4,
        allowed_labels=("KV_CACHE", "LONG_MEMORY", "BRIDGE"),
        bridge_regression_slugs=(),
    )


def result(candidate, label: str, confidence: str = "HIGH", **overrides: str) -> dict[str, str]:
    values = {
        "schema_version": ROUTER_RESULT_SCHEMA_VERSION,
        "job_id": expected_job_id(
            candidate.slug, candidate.note_sha256, PROMPT_SHA256
        ),
        "slug": candidate.slug,
        "note_sha256": candidate.note_sha256,
        "prompt_sha256": PROMPT_SHA256,
        "label": label,
        "core_mechanism": f"{candidate.slug} mechanism",
        "scope_reason": f"{candidate.slug} scope",
        "evidence_locator": "note:2",
        "confidence": confidence,
    }
    values.update(overrides)
    return values


def candidate_map(candidates) -> dict[str, object]:
    return {candidate.slug: candidate for candidate in candidates}


def valid_fixture_results(candidates) -> list[dict[str, str]]:
    by_slug = candidate_map(candidates)
    labels = {
        "paper-a": ("KV_CACHE", "HIGH"),
        "paper-bridge": ("BRIDGE", "HIGH"),
        "paper-keyword": ("OTHER", "HIGH"),
        "paper-kv": ("KV_CACHE", "HIGH"),
        "paper-memory": ("LONG_MEMORY", "MEDIUM"),
    }
    return [result(by_slug[slug], *labels[slug]) for slug in sorted(labels)]


def test_enumerate_excludes_audits_deduplicates_and_preserves_provenance() -> None:
    candidates = enumerate_candidates(config())

    assert [candidate.slug for candidate in candidates] == [
        "paper-a",
        "paper-bridge",
        "paper-keyword",
        "paper-kv",
        "paper-memory",
    ]
    paper_a = candidate_map(candidates)["paper-a"]
    assert paper_a.source_lists == (
        str((FIXTURES / "candidates" / "list-one.txt").resolve()),
        str((FIXTURES / "candidates" / "list-two.txt").resolve()),
    )
    assert paper_a.note_sha256 == hashlib.sha256(
        (FIXTURES / "notes" / "paper-a.md").read_bytes()
    ).hexdigest()


def test_enumeration_rejects_missing_notes_conflicts_and_missing_controls(tmp_path: Path) -> None:
    missing_note = tmp_path / "notes" / "missing.md"
    with pytest.raises(ValueError, match="missing base note"):
        enumerate_candidates(scoped_config(tmp_path, [missing_note]))

    first = tmp_path / "notes" / "paper-a.md"
    second = tmp_path / "papers" / "paper-a.md"
    first.write_text("first paper", encoding="utf-8")
    second.write_text("different paper", encoding="utf-8")
    with pytest.raises(ValueError, match="different base files"):
        enumerate_candidates(scoped_config(tmp_path, [first, second]))

    with pytest.raises(ValueError, match="missing configured bridge regression.*absent-control"):
        enumerate_candidates(config(bridge_regression_slugs=("paper-bridge", "absent-control")))


def test_enumeration_normalizes_mixed_case_filename_slug(tmp_path: Path) -> None:
    note = tmp_path / "notes" / "paper-MiXeD.md"
    note.parent.mkdir()
    note.write_text("mixed-case source identifier", encoding="utf-8")

    candidates = enumerate_candidates(
        replace(
            scoped_config(tmp_path, [note]),
            bridge_regression_slugs=("paper-mixed",),
        )
    )

    assert [candidate.slug for candidate in candidates] == ["paper-mixed"]


def test_mixed_case_audit_is_excluded_from_candidates_and_jobs(tmp_path: Path) -> None:
    notes_root = tmp_path / "notes"
    notes_root.mkdir()
    base = notes_root / "paper-base.md"
    audit = notes_root / "paper-base.AUDIT.MD"
    base.write_text("base note", encoding="utf-8")
    audit.write_text("audit must not be routed", encoding="utf-8")
    candidates = enumerate_candidates(scoped_config(tmp_path, [audit, base]))

    output = write_router_jobs(candidates, tmp_path / "run", prompt_path=PROMPT)

    assert [candidate.slug for candidate in candidates] == ["paper-base"]
    assert [record["slug"] for record in read_jsonl(output)] == ["paper-base"]


def test_candidate_and_list_paths_are_confined_to_configured_roots(tmp_path: Path) -> None:
    outside_note = tmp_path / "outside.md"
    outside_note.write_text("outside", encoding="utf-8")
    bounded = scoped_config(tmp_path, [outside_note])
    with pytest.raises(ValueError, match="outside configured candidate roots"):
        enumerate_candidates(bounded)

    outside_list = tmp_path / "outside-list.txt"
    outside_list.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate list.*corpus root"):
        enumerate_candidates(
            config(candidate_lists=(outside_list,), bridge_regression_slugs=())
        )


def test_legacy_ledger_entry_is_explicitly_excluded(tmp_path: Path) -> None:
    note = tmp_path / "notes" / "paper.md"
    note.parent.mkdir()
    note.write_text("paper", encoding="utf-8")
    bounded = scoped_config(tmp_path, [])
    bounded.candidate_lists[0].write_text(
        f"{bounded.legacy_ledger}\n{note}\n", encoding="utf-8"
    )

    candidates = enumerate_candidates(bounded)

    assert [candidate.slug for candidate in candidates] == ["paper"]


def test_candidate_symlink_escape_is_rejected_when_supported(tmp_path: Path) -> None:
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside = outside_dir / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    notes_root = tmp_path / "notes"
    notes_root.mkdir()
    link = notes_root / "escape"
    directory_link_or_skip(link, outside_dir)

    with pytest.raises(ValueError, match="outside configured candidate roots"):
        enumerate_candidates(scoped_config(tmp_path, [link / outside.name]))


@pytest.mark.parametrize("kind", ["outside", "directory", "junction_escape"])
def test_candidate_list_is_preflighted_before_hash_or_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    notes = tmp_path / "notes"
    papers = tmp_path / "papers"
    legacy = papers / "_ideas" / "ledger.md"
    notes.mkdir()
    legacy.parent.mkdir(parents=True)
    legacy.write_text("legacy", encoding="utf-8")
    if kind == "outside":
        candidate_list = tmp_path / "outside.txt"
        candidate_list.write_text("unused.md\n", encoding="utf-8")
    elif kind == "directory":
        candidate_list = papers / "lists"
        candidate_list.mkdir()
    else:
        outside = tmp_path / "outside-dir"
        outside.mkdir()
        (outside / "list.txt").write_text("unused.md\n", encoding="utf-8")
        link = papers / "escape"
        directory_link_or_skip(link, outside)
        candidate_list = link / "list.txt"
    bounded = CorpusRouterConfig(
        notes_root=notes,
        candidate_lists=(candidate_list,),
        legacy_ledger=legacy,
        target_min=1,
        target_max=1,
        allowed_labels=("KV_CACHE",),
        bridge_regression_slugs=(),
    )
    touched: list[str] = []

    def forbidden_hash(path: Path) -> str:
        touched.append(f"hash:{path}")
        raise AssertionError("candidate list was hashed before preflight")

    def forbidden_read(path: Path, *args, **kwargs) -> str:
        touched.append(f"read:{path}")
        raise AssertionError("candidate list was read before preflight")

    monkeypatch.setattr(corpus_module, "sha256_file", forbidden_hash)
    monkeypatch.setattr(Path, "read_text", forbidden_read)
    with pytest.raises(ValueError, match="candidate list"):
        enumerate_candidates(bounded)
    assert touched == []


def test_default_config_resolution_discovers_coherent_ancestor_for_nested_worktree(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    checkout = workspace / "idea_factory"
    nested_repo = checkout / ".worktrees" / "idea-factory-v1"
    config_path = nested_repo / "config" / "pilot.json"
    notes = workspace / "notes"
    papers = workspace / "papers"
    (papers / "_ideas").mkdir(parents=True)
    notes.mkdir()
    (papers / "list.txt").write_text("unused.md\n", encoding="utf-8")
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "notes_root": "../notes",
                "candidate_lists": ["../papers/list.txt"],
                "legacy_ledger": "../papers/_ideas/ledger.md",
                "target_min": 1,
                "target_max": 2,
                "allowed_labels": ["KV_CACHE"],
                "bridge_regression_slugs": [],
            }
        ),
        encoding="utf-8",
    )

    loaded = CorpusRouterConfig.from_json(config_path)

    assert loaded.notes_root == notes.resolve()
    assert loaded.candidate_lists == ((papers / "list.txt").resolve(),)
    assert loaded.legacy_ledger == (papers / "_ideas" / "ledger.md").resolve()


def test_default_config_resolution_fails_without_coherent_ancestor(tmp_path: Path) -> None:
    config_path = tmp_path / "repo" / "config" / "pilot.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "notes_root": "../notes",
                "candidate_lists": ["../papers/missing.txt"],
                "legacy_ledger": "../papers/_ideas/ledger.md",
                "target_min": 1,
                "target_max": 2,
                "allowed_labels": ["KV_CACHE"],
                "bridge_regression_slugs": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no coherent config base"):
        CorpusRouterConfig.from_json(config_path)


def test_explicit_repo_root_resolution_remains_available(tmp_path: Path) -> None:
    config_path = tmp_path / "idea_factory" / "config" / "pilot.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "notes_root": "../notes",
                "candidate_lists": ["../papers/list.txt"],
                "legacy_ledger": "../papers/ledger.md",
                "target_min": 1,
                "target_max": 2,
                "allowed_labels": ["KV_CACHE"],
                "bridge_regression_slugs": [],
            }
        ),
        encoding="utf-8",
    )

    loaded = CorpusRouterConfig.from_json(
        config_path, repo_root=tmp_path / "workspace" / "idea_factory"
    )

    assert loaded.notes_root == (tmp_path / "workspace" / "notes").resolve()
    assert loaded.candidate_lists == (
        (tmp_path / "workspace" / "papers" / "list.txt").resolve(),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_min", True),
        ("target_min", 1.5),
        ("target_max", False),
        ("target_max", 4.5),
    ],
)
def test_config_rejects_non_exact_integer_bounds(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="exact positive integers"):
        config(**{field: value})


@pytest.mark.parametrize(
    "overrides",
    [
        {"allowed_labels": ("KV_CACHE", "KV_CACHE")},
        {"allowed_labels": ("kv_cache",)},
        {"bridge_regression_slugs": ("Paper-Bridge", "paper-bridge")},
    ],
)
def test_config_rejects_duplicate_or_noncanonical_labels_and_slugs(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="allowed_labels|bridge_regression_slugs"):
        config(**overrides)


def test_config_rejects_duplicate_candidate_lists_after_resolution() -> None:
    candidate_list = FIXTURES / "candidates" / "list-one.txt"

    with pytest.raises(ValueError, match="candidate_lists.*unique"):
        config(candidate_lists=(candidate_list, candidate_list.parent / "." / candidate_list.name))


@pytest.mark.parametrize(
    "overrides",
    [
        {"candidate_lists": "abc"},
        {"candidate_lists": (" ",)},
        {"bridge_regression_slugs": "abc"},
    ],
)
def test_config_rejects_scalar_or_blank_path_and_slug_collections(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="candidate_lists|bridge_regression_slugs"):
        config(**overrides)


def test_json_config_rejects_scalar_bridge_collection(tmp_path: Path) -> None:
    config_path = tmp_path / "config" / "pilot.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "notes_root": "../notes",
                "candidate_lists": ["../papers/list.txt"],
                "legacy_ledger": "../papers/_ideas/ledger.md",
                "target_min": 1,
                "target_max": 2,
                "allowed_labels": ["KV_CACHE"],
                "bridge_regression_slugs": "abc",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="bridge_regression_slugs"):
        CorpusRouterConfig.from_json(config_path, repo_root=tmp_path / "idea_factory")


def test_router_jobs_embed_bound_note_prompt_and_schema_without_touching_sources(
    tmp_path: Path,
) -> None:
    candidates = enumerate_candidates(config())
    source_before = (FIXTURES / "candidates" / "list-one.txt").read_bytes()

    output = write_router_jobs(candidates, tmp_path, prompt_path=PROMPT)
    first = output.read_bytes()
    write_router_jobs(candidates, tmp_path, prompt_path=PROMPT)
    records = read_jsonl(output)

    assert output == tmp_path / "corpus" / "router_jobs.jsonl"
    assert output.read_bytes() == first
    assert len(records) == len(candidates)
    first_candidate = candidate_map(candidates)[records[0]["slug"]]
    assert records[0]["schema_version"] == ROUTER_JOB_SCHEMA_VERSION
    assert records[0]["result_schema_version"] == ROUTER_RESULT_SCHEMA_VERSION
    assert records[0]["note_text"] == first_candidate.note_path.read_bytes().decode("utf-8")
    assert records[0]["prompt_text"] == PROMPT.read_bytes().decode("utf-8")
    assert records[0]["prompt_path"] == str(PROMPT.resolve())
    assert records[0]["prompt_sha256"] == PROMPT_SHA256
    assert records[0]["prompt_version"] == PROMPT_VERSION
    assert records[0]["required_output_schema"]["additionalProperties"] is False
    assert records[0]["source_list_bindings"]
    assert records[0]["candidate_list_manifest"] == sorted(
        records[0]["candidate_list_manifest"], key=lambda item: item["path"]
    )
    assert all(Path(item["path"]).is_file() and len(item["sha256"]) == 64 for item in records[0]["candidate_list_manifest"])
    assert records[0]["job_id"] == expected_job_id(
        first_candidate.slug, first_candidate.note_sha256, PROMPT_SHA256
    )
    assert (FIXTURES / "candidates" / "list-one.txt").read_bytes() == source_before


def test_router_job_output_rejects_corpus_symlink_escape_without_external_changes(
    tmp_path: Path,
) -> None:
    candidates = enumerate_candidates(config())
    run_dir = tmp_path / "run"
    outside = tmp_path / "outside"
    run_dir.mkdir()
    outside.mkdir()
    external_jobs = outside / "router_jobs.jsonl"
    external_selection = outside / "selection_manifest.jsonl"
    external_rejected = outside / "rejected_manifest.jsonl"
    for path in (external_jobs, external_selection, external_rejected):
        path.write_text(f"sentinel:{path.name}", encoding="utf-8")
    directory_link_or_skip(run_dir / "corpus", outside)

    with pytest.raises(ValueError, match="corpus output directory.*active run"):
        write_router_jobs(candidates, run_dir, prompt_path=PROMPT)

    assert external_jobs.read_text(encoding="utf-8") == "sentinel:router_jobs.jsonl"
    assert external_selection.read_text(encoding="utf-8") == "sentinel:selection_manifest.jsonl"
    assert external_rejected.read_text(encoding="utf-8") == "sentinel:rejected_manifest.jsonl"


def test_router_job_rerun_invalidates_downstream_manifests_even_on_failure(
    tmp_path: Path,
) -> None:
    candidates = enumerate_candidates(config())
    ingested = ingest_router_results(
        candidates,
        valid_fixture_results(candidates),
        prompt_sha256=PROMPT_SHA256,
    )
    write_router_jobs(candidates, tmp_path, prompt_path=PROMPT)
    write_router_jobs(candidates, tmp_path, prompt_path=PROMPT)
    select_pilot(config(), ingested, tmp_path)
    selection_path = tmp_path / "corpus" / "selection_manifest.jsonl"
    rejected_path = tmp_path / "corpus" / "rejected_manifest.jsonl"
    changed_prompt = tmp_path / "changed-prompt.md"
    changed_prompt.write_text(PROMPT.read_text(encoding="utf-8") + "\nchanged", encoding="utf-8")

    write_router_jobs(candidates, tmp_path, prompt_path=changed_prompt)

    assert not selection_path.exists()
    assert not rejected_path.exists()

    write_router_jobs(candidates, tmp_path, prompt_path=PROMPT)
    select_pilot(config(), ingested, tmp_path)
    with pytest.raises(ValueError, match="router prompt"):
        write_router_jobs(candidates, tmp_path, prompt_path=tmp_path / "missing.md")

    assert not selection_path.exists()
    assert not rejected_path.exists()


def test_router_prompt_requires_exact_bound_json_result() -> None:
    prompt = PROMPT.read_text(encoding="utf-8")

    assert '"schema_version":"idea_factory.corpus_router_result.v2"' in prompt
    assert '"job_id":"string"' in prompt
    assert '"note_sha256":"string"' in prompt
    assert '"prompt_sha256":"string"' in prompt
    assert "Output exactly one JSON object and no prose or Markdown" in prompt


def test_ingestion_requires_exactly_one_strict_bound_result_per_candidate() -> None:
    candidates = enumerate_candidates(config())
    valid = valid_fixture_results(candidates)

    ingested = ingest_router_results(candidates, valid, prompt_sha256=PROMPT_SHA256)
    assert ingested[0].candidate.note_sha256
    assert ingested[0].result.job_id == valid[0]["job_id"]
    assert ingested[0].raw_result_sha256 == hashlib.sha256(
        json.dumps(valid[0], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    with pytest.raises(ValueError, match="exactly one result"):
        ingest_router_results(candidates, valid[:-1], prompt_sha256=PROMPT_SHA256)
    unexpected = type(
        "Candidate",
        (),
        {"slug": "not-a-candidate", "note_sha256": "f" * 64},
    )()
    with pytest.raises(ValueError, match="unexpected slug"):
        ingest_router_results(
            candidates,
            valid + [result(unexpected, "OTHER")],
            prompt_sha256=PROMPT_SHA256,
        )
    with pytest.raises(ValueError, match="duplicate slug"):
        ingest_router_results(
            candidates, valid + [valid[0]], prompt_sha256=PROMPT_SHA256
        )
    bad = [*valid]
    bad[0] = {**bad[0], "unexpected": "field"}
    with pytest.raises(ValueError, match="invalid router result"):
        ingest_router_results(candidates, bad, prompt_sha256=PROMPT_SHA256)


def test_ingestion_owns_an_immutable_revalidated_result_copy() -> None:
    candidates = enumerate_candidates(config())
    supplied = RouterResult.model_validate(valid_fixture_results(candidates)[0])
    remaining = valid_fixture_results(candidates)[1:]

    ingested = ingest_router_results(
        candidates, [supplied, *remaining], prompt_sha256=PROMPT_SHA256
    )
    owned = next(item for item in ingested if item.candidate.slug == supplied.slug)
    original_hash = owned.raw_result_sha256

    assert owned.result is not supplied
    with pytest.raises(ValidationError):
        owned.result.core_mechanism = "mutated"
    object.__setattr__(supplied, "core_mechanism", "bypassed mutation")

    assert owned.result.core_mechanism != supplied.core_mechanism
    assert owned.raw_result_sha256 == original_hash


@pytest.mark.parametrize("binding", ["note_sha256", "prompt_sha256", "job_id"])
def test_ingestion_rejects_stale_or_cross_run_result_binding(binding: str) -> None:
    candidates = enumerate_candidates(config())
    valid = valid_fixture_results(candidates)
    valid[0] = {**valid[0], binding: "f" * 64}

    with pytest.raises(ValueError, match="binding mismatch"):
        ingest_router_results(candidates, valid, prompt_sha256=PROMPT_SHA256)


def test_selection_preserves_valid_bridge_and_rejects_other_and_low(tmp_path: Path) -> None:
    candidates = enumerate_candidates(config())
    by_slug = candidate_map(candidates)
    routed = valid_fixture_results(candidates)
    for index, item in enumerate(routed):
        if item["slug"] == "paper-kv":
            routed[index] = result(by_slug["paper-kv"], "KV_CACHE", "LOW")
    ingested = ingest_router_results(candidates, routed, prompt_sha256=PROMPT_SHA256)
    write_router_jobs(candidates, tmp_path, prompt_path=PROMPT)

    selected, rejected = select_pilot(config(), ingested, tmp_path)

    assert {item.candidate.slug for item in selected} == {
        "paper-a",
        "paper-bridge",
        "paper-memory",
    }
    assert {item.reason_code for item in rejected} == {"LABEL_OTHER", "CONFIDENCE_LOW"}
    selection_records = read_jsonl(tmp_path / "corpus" / "selection_manifest.jsonl")
    rejected_records = read_jsonl(tmp_path / "corpus" / "rejected_manifest.jsonl")
    assert {record["reason_code"] for record in selection_records} == {"ELIGIBLE"}
    assert all(record["job_id"] and record["result_schema_version"] for record in selection_records)
    provenance = [*selection_records, *rejected_records]
    assert {record["allowed_note_root"] for record in provenance} == {str(config().notes_root)}
    assert {record["allowed_papers_root"] for record in provenance} == {str(config().legacy_ledger.parent.parent)}
    assert {record["legacy_ledger_path"] for record in provenance} == {str(config().legacy_ledger)}
    assert {record["router_jobs_sha256"] for record in provenance} == {
        hashlib.sha256((tmp_path / "corpus" / "router_jobs.jsonl").read_bytes()).hexdigest()
    }


def test_selection_is_stratified_and_independent_of_result_order(tmp_path: Path) -> None:
    base = [
        {"slug": f"kv-{index}", "label": "KV_CACHE", "confidence": "HIGH"}
        for index in range(5)
    ] + [
        {"slug": "memory-0", "label": "LONG_MEMORY", "confidence": "HIGH"},
        {"slug": "bridge-0", "label": "BRIDGE", "confidence": "HIGH"},
    ]
    candidates = [
        type(
            "Candidate",
            (),
            {
                "slug": value["slug"],
                "note_path": Path(f"/{value['slug']}.md"),
                "source_lists": ("fixture",),
                "note_sha256": "a" * 64,
            },
        )()
        for value in reversed(base)
    ]
    results = [
        RouterResult.model_validate(result(candidate_map(candidates)[value["slug"]], value["label"], value["confidence"]))
        for value in base
    ]
    ingested = ingest_router_results(candidates, results, prompt_sha256=PROMPT_SHA256)
    selected = _choose_stratified(
        ingested,
        config(target_max=3, bridge_regression_slugs=("bridge-0",)),
    )

    assert {item.candidate.slug for item in selected} == {"bridge-0", "memory-0", "kv-0"}


def test_selection_fails_for_missing_or_invalid_bridge_and_below_minimum(tmp_path: Path) -> None:
    candidates = enumerate_candidates(config())
    by_slug = candidate_map(candidates)
    routed = [
        result(by_slug["paper-a"], "OTHER"),
        result(by_slug["paper-bridge"], "BRIDGE", "LOW"),
        result(by_slug["paper-keyword"], "OTHER"),
        result(by_slug["paper-kv"], "OTHER"),
        result(by_slug["paper-memory"], "OTHER"),
    ]
    ingested = ingest_router_results(candidates, routed, prompt_sha256=PROMPT_SHA256)
    write_router_jobs(candidates, tmp_path, prompt_path=PROMPT)
    with pytest.raises(ValueError, match="bridge regression"):
        select_pilot(config(target_min=1), ingested, tmp_path)

    with pytest.raises(ValueError, match="missing configured bridge regression"):
        select_pilot(
            config(bridge_regression_slugs=("paper-bridge", "absent-control")),
            ingested,
            tmp_path,
        )

    valid = ingest_router_results(
        candidates,
        [result(candidate, "OTHER") for candidate in candidates],
        prompt_sha256=PROMPT_SHA256,
    )
    with pytest.raises(ValueError, match="target_min"):
        select_pilot(config(target_min=1, bridge_regression_slugs=()), valid, tmp_path)


@pytest.mark.parametrize(
    ("failure_kind", "error_match"),
    [
        ("target_min", "target_min"),
        ("missing_control", "missing configured bridge regression"),
        ("invalid_control", "bridge regression candidate is invalid"),
    ],
)
def test_failed_selection_rerun_removes_only_stale_owned_manifests(
    tmp_path: Path, failure_kind: str, error_match: str
) -> None:
    candidates = enumerate_candidates(config())
    successful = ingest_router_results(
        candidates,
        valid_fixture_results(candidates),
        prompt_sha256=PROMPT_SHA256,
    )
    jobs_path = write_router_jobs(candidates, tmp_path, prompt_path=PROMPT)
    corpus_dir = jobs_path.parent
    jobs_before = jobs_path.read_bytes()
    source_path = FIXTURES / "candidates" / "list-one.txt"
    source_before = source_path.read_bytes()

    select_pilot(config(), successful, tmp_path)
    selection_path = corpus_dir / "selection_manifest.jsonl"
    rejected_path = corpus_dir / "rejected_manifest.jsonl"
    assert selection_path.is_file()
    assert rejected_path.is_file()

    failing_config = config()
    failing_ingested = successful
    if failure_kind == "target_min":
        failing_config = config(target_min=5, target_max=5)
    elif failure_kind == "missing_control":
        failing_config = config(
            bridge_regression_slugs=("paper-bridge", "absent-control")
        )
    else:
        by_slug = candidate_map(candidates)
        invalid_results = valid_fixture_results(candidates)
        bridge_index = next(
            index
            for index, item in enumerate(invalid_results)
            if item["slug"] == "paper-bridge"
        )
        invalid_results[bridge_index] = result(
            by_slug["paper-bridge"], "BRIDGE", "LOW"
        )
        failing_ingested = ingest_router_results(
            candidates,
            invalid_results,
            prompt_sha256=PROMPT_SHA256,
        )

    with pytest.raises(ValueError, match=error_match):
        select_pilot(failing_config, failing_ingested, tmp_path)

    assert not selection_path.exists()
    assert not rejected_path.exists()
    assert jobs_path.read_bytes() == jobs_before
    assert source_path.read_bytes() == source_before


def test_second_manifest_publish_failure_invalidates_bundle_and_cleans_temps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = enumerate_candidates(config())
    ingested = ingest_router_results(
        candidates,
        valid_fixture_results(candidates),
        prompt_sha256=PROMPT_SHA256,
    )
    write_router_jobs(candidates, tmp_path, prompt_path=PROMPT)
    real_replace = artifacts_module.os.replace
    replace_calls = 0

    def fail_second_replace(source, destination):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("simulated second publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(artifacts_module.os, "replace", fail_second_replace)

    with pytest.raises(OSError, match="second publish failure"):
        select_pilot(config(), ingested, tmp_path)

    corpus_dir = tmp_path / "corpus"
    assert not (corpus_dir / "selection_manifest.jsonl").exists()
    assert not (corpus_dir / "rejected_manifest.jsonl").exists()
    assert not list(corpus_dir.glob("*.tmp"))
    assert not list(corpus_dir.glob(".*.tmp"))


@pytest.mark.parametrize("mutation", ["label", "confidence", "reason", "result_hash"])
def test_selection_replay_rejects_decision_field_mutation(tmp_path: Path, mutation: str) -> None:
    candidates = enumerate_candidates(config())
    write_router_jobs(candidates, tmp_path, prompt_path=PROMPT)
    ingested = ingest_router_results(candidates, valid_fixture_results(candidates), prompt_sha256=PROMPT_SHA256)
    select_pilot(config(), ingested, tmp_path)
    selection = tmp_path / "corpus" / "selection_manifest.jsonl"
    rows = read_jsonl(selection)
    field_values = {"label": "OTHER", "confidence": "LOW", "reason": "TARGET_MAX", "result_hash": "f" * 64}
    field_names = {"label": "label", "confidence": "confidence", "reason": "reason_code", "result_hash": "raw_result_sha256"}
    rows[0][field_names[mutation]] = field_values[mutation]
    selection.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="selection|decision|manifest"):
        validate_corpus_selection_bundle(selection)


@pytest.mark.parametrize("mutation", ["missing_results", "changed_results", "changed_policy"])
def test_selection_replay_requires_bound_router_results_and_policy(tmp_path: Path, mutation: str) -> None:
    candidates = enumerate_candidates(config())
    write_router_jobs(candidates, tmp_path, prompt_path=PROMPT)
    ingested = ingest_router_results(candidates, valid_fixture_results(candidates), prompt_sha256=PROMPT_SHA256)
    select_pilot(config(), ingested, tmp_path)
    corpus = tmp_path / "corpus"
    target = corpus / ("selection_policy.jsonl" if mutation == "changed_policy" else "router_results.jsonl")
    if mutation == "missing_results":
        target.unlink()
    elif mutation == "changed_policy":
        record = read_jsonl(target)[0]
        record["target_max"] += 1
        target.write_text(json.dumps(record) + "\n", encoding="utf-8")
    else:
        target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="router results|policy|bundle|snapshot|replayed decisions"):
        validate_corpus_selection_bundle(corpus / "selection_manifest.jsonl")
