from __future__ import annotations

import json
import hashlib
import csv
from pathlib import Path
import subprocess

import pytest

from idea_factory.cards import emit_card_jobs, ingest_card_results
from test_cards import _card, _result, _selection_bundle


def _bundle(tmp_path: Path, names: tuple[str, ...] = ("alpha", "beta")) -> tuple[Path, list[dict[str, object]]]:
    run, selection, prompt = _selection_bundle(tmp_path, names)
    jobs_path = emit_card_jobs(selection, prompt, run / "cards" / "card_jobs.jsonl")
    from idea_factory.artifacts import read_jsonl

    return run, read_jsonl(jobs_path)


def test_mechanical_normalization_merges_only_exact_scope_and_writes_owned_bundle(tmp_path: Path) -> None:
    from idea_factory.landscape import build_landscape

    run, jobs = _bundle(tmp_path)
    first = _card(jobs[0], "one", assumption={"text": " Importance   Score ", "status": "REPORTED"})
    second = _card(jobs[1], "two", assumption={"text": "importance score", "status": "INFERRED"})
    second["scope"]["time_horizon"] = "days"
    ingest_card_results(
        run / "cards" / "card_jobs.jsonl",
        [_result(jobs[1], [second]), _result(jobs[0], [first])],
    )

    outputs = build_landscape(run)

    assert set(outputs) == {"assumptions", "failures", "mechanisms", "evaluations", "edges", "assignments"}
    entries = json.loads((run / "landscape" / "assumptions.json").read_text(encoding="utf-8"))["entries"]
    assert len([entry for entry in entries if entry["normalized_text"] == "importance score"]) == 2
    assert all(path.is_file() for path in outputs.values())


@pytest.mark.parametrize("field", ["object", "time_horizon", "setting"])
def test_reviewed_v2_cluster_scope_rejects_blank_dimensions(tmp_path: Path, field: str) -> None:
    from idea_factory.landscape import build_landscape, canonical_scope_key

    run, jobs = _bundle(tmp_path)
    card = _card(jobs[0], "one")
    ingest_card_results(run / "cards" / "card_jobs.jsonl", [_result(jobs[0], [card]), _result(jobs[1], [])])
    cluster_scope = {"object": "retained unit", "time_horizon": "future use", "setting": "retention"}
    cluster_scope[field] = ""
    row = _v2_override(
        "one", "cache stays warm", canonical_scope_key(card["scope"]),
        normalized="warm cache", cluster_scope=cluster_scope,
    )

    with pytest.raises(ValueError, match="reviewed assignment|cluster|nonblank"):
        build_landscape(run, _v2_review_artifact(tmp_path / "blank-scope.json", [row]))


def _review_artifact(path: Path, rows: list[dict[str, object]]) -> Path:
    body = {
        "schema_version": "idea_factory.concept_assignments.v1",
        "assignments": rows,
    }
    body["artifact_sha256"] = hashlib.sha256(
        json.dumps(sorted(rows, key=lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    return path


def _override(card_id: str, raw: str, scope_key: str, *, normalized: str, status: str = "APPROVED") -> dict[str, object]:
    return {
        "schema_version": "idea_factory.concept_assignment.v1", "dimension": "assumptions",
        "facet": "assumption", "raw_text": raw, "normalized_text": normalized,
        "card_id": card_id, "scope_key": scope_key, "merge_reason": "reviewed equivalence",
        "disputed": status == "DISPUTED", "reviewed": True, "review_status": status,
        "reviewer_reason": "recorded review",
    }


def _v2_override(
    card_id: str,
    raw: str,
    source_scope_key: str,
    *,
    normalized: str,
    cluster_scope: dict[str, str],
) -> dict[str, object]:
    return {
        "schema_version": "idea_factory.concept_assignment.v2",
        "dimension": "assumptions",
        "facet": "assumption",
        "raw_text": raw,
        "normalized_text": normalized,
        "card_id": card_id,
        "source_scope_key": source_scope_key,
        "cluster_scope": cluster_scope,
        "merge_reason": "reviewed scientific equivalence within a canonical coarse scope",
        "disputed": False,
        "reviewed": True,
        "review_status": "APPROVED",
        "reviewer_reason": "raw evidence scopes differ, but the reviewed opportunity basis is the same",
    }


def _v2_review_artifact(path: Path, rows: list[dict[str, object]]) -> Path:
    body = {
        "schema_version": "idea_factory.concept_assignments.v2",
        "assignments": rows,
    }
    body["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            sorted(rows, key=lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    return path


def test_mechanical_case_whitespace_merges_but_synonyms_do_not(tmp_path: Path) -> None:
    from idea_factory.landscape import build_landscape

    run, jobs = _bundle(tmp_path)
    first = _card(jobs[0], "one", assumption={"text": "Importance   Score", "status": "REPORTED"})
    second = _card(jobs[1], "two", assumption={"text": "significance score", "status": "INFERRED"})
    ingest_card_results(run / "cards" / "card_jobs.jsonl", [_result(jobs[0], [first]), _result(jobs[1], [second])])
    build_landscape(run)
    entries = json.loads((run / "landscape" / "assumptions.json").read_text(encoding="utf-8"))["entries"]
    assert {entry["normalized_text"] for entry in entries} >= {"importance score", "significance score"}

    second["assumption"] = {"text": " importance score ", "status": "INFERRED"}
    ingest_card_results(run / "cards" / "card_jobs.jsonl", [_result(jobs[1], [second]), _result(jobs[0], [first])])
    build_landscape(run)
    entries = json.loads((run / "landscape" / "assumptions.json").read_text(encoding="utf-8"))["entries"]
    merged = [entry for entry in entries if entry["normalized_text"] == "importance score"]
    assert len(merged) == 1 and merged[0]["source_card_ids"] == ["one", "two"]


def test_approved_override_merges_only_exact_scope_and_disputed_never_merges(tmp_path: Path) -> None:
    from idea_factory.landscape import build_landscape, canonical_scope_key

    run, jobs = _bundle(tmp_path)
    first = _card(jobs[0], "one", assumption={"text": "importance score", "status": "REPORTED"})
    second = _card(jobs[1], "two", assumption={"text": "significance score", "status": "OBSERVED"})
    ingest_card_results(run / "cards" / "card_jobs.jsonl", [_result(jobs[0], [first]), _result(jobs[1], [second])])
    scope_key = canonical_scope_key(first["scope"])
    review = _review_artifact(tmp_path / "review.json", [_override("two", "significance score", scope_key, normalized="importance score")])
    build_landscape(run, review)
    entries = json.loads((run / "landscape" / "assumptions.json").read_text(encoding="utf-8"))["entries"]
    assert len([entry for entry in entries if entry["normalized_text"] == "importance score"]) == 1

    disputed = _review_artifact(tmp_path / "disputed.json", [_override("two", "significance score", scope_key, normalized="importance score", status="DISPUTED")])
    build_landscape(run, disputed)
    entries = json.loads((run / "landscape" / "assumptions.json").read_text(encoding="utf-8"))["entries"]
    labels = {entry["normalized_text"] for entry in entries}
    assert {"importance score", "significance score"} <= labels
    assert any(entry["has_disputed_assignments"] for entry in entries)

    second["scope"]["time_horizon"] = "days"
    ingest_card_results(run / "cards" / "card_jobs.jsonl", [_result(jobs[0], [first]), _result(jobs[1], [second])])
    with pytest.raises(ValueError, match="stale or orphan"):
        build_landscape(run, review)


def test_reviewed_v2_cluster_scope_preserves_raw_scope_and_enables_cross_scope_basis(tmp_path: Path) -> None:
    from idea_factory.artifacts import read_jsonl
    from idea_factory.landscape import build_landscape, canonical_scope_key, validate_landscape_bundle
    from idea_factory.opportunities import emit_mining_jobs

    run, jobs = _bundle(tmp_path, ("alpha", "beta", "gamma"))
    raw_scopes = [
        {"object": "token cache", "time_horizon": "one decode", "setting": "serving"},
        {"object": "episodic record", "time_horizon": "many sessions", "setting": "agent"},
        {"object": "visual memory", "time_horizon": "many clips", "setting": "streaming"},
    ]
    assumptions = ["token utility stays fixed", "memory value is stationary", "visual evidence value does not drift"]
    cards = []
    for index, job in enumerate(jobs):
        cards.append(_card(job, f"card-{index}", assumption={"text": assumptions[index], "status": "INFERRED"}, scope=raw_scopes[index]))
    ingest_card_results(
        run / "cards" / "card_jobs.jsonl",
        [_result(job, [card]) for job, card in zip(jobs, cards, strict=True)],
    )

    cluster_scope = {
        "object": "retained information unit",
        "time_horizon": "future-dependent utility",
        "setting": "adaptive retention",
    }
    rows = [
        _v2_override(
            card["card_id"],
            assumptions[index],
            canonical_scope_key(raw_scopes[index]),
            normalized="static utility assumption",
            cluster_scope=cluster_scope,
        )
        for index, card in enumerate(cards)
    ]
    review = _v2_review_artifact(tmp_path / "review-v2.json", rows)

    build_landscape(run, review)
    bundle = validate_landscape_bundle(run)
    merged = [
        entry
        for entry in bundle.maps["assumptions"]["entries"]
        if entry["normalized_text"] == "static utility assumption"
    ]
    assert len(merged) == 1
    assert merged[0]["scope"] == cluster_scope
    assert merged[0]["source_card_ids"] == ["card-0", "card-1", "card-2"]
    edges = [
        edge
        for edge in bundle.edges
        if edge["normalized_text"] == "static utility assumption"
    ]
    assert len({edge["scope_key"] for edge in edges}) == 3
    assert {json.loads(edge["scope"])["object"] for edge in edges} == {
        "token cache", "episodic record", "visual memory",
    }
    assert len({edge["cluster_scope_key"] for edge in edges}) == 1

    prompt = tmp_path / "opportunity.md"
    prompt.write_text("strict opportunity json", encoding="utf-8")
    jobs_out = read_jsonl(emit_mining_jobs(run, prompt))
    matching = [
        row
        for row in jobs_out
        if row["cluster_basis"]["normalized_text"] == "static utility assumption"
    ]
    assert len(matching) == 8
    assert all(row["card_ids"] == ["card-0", "card-1", "card-2"] for row in matching)
    assert all(row["neighbor_ids"] for row in matching)
    assert all(
        any(
            "SAME_CARD_CONTEXT_OTHER_SCOPE" in neighbor["relations"]
            for neighbor in row["neighbor_summaries"]
        )
        for row in matching
    )


def test_v2_assignment_artifact_replay_rejects_coordinated_scope_and_map_tamper(tmp_path: Path) -> None:
    from idea_factory.landscape import build_landscape, canonical_scope_key, validate_landscape_bundle

    run, jobs = _bundle(tmp_path)
    cards = [
        _card(jobs[0], "one", assumption={"text": "utility is fixed", "status": "INFERRED"}),
        _card(jobs[1], "two", assumption={"text": "value is stationary", "status": "INFERRED"}),
    ]
    cards[1]["scope"]["time_horizon"] = "days"
    ingest_card_results(
        run / "cards" / "card_jobs.jsonl",
        [_result(job, [card]) for job, card in zip(jobs, cards, strict=True)],
    )
    reviewed_scope = {"object": "retained unit", "time_horizon": "future use", "setting": "retention"}
    review = _v2_review_artifact(tmp_path / "review-v2.json", [
        _v2_override(card["card_id"], card["assumption"]["text"], canonical_scope_key(card["scope"]), normalized="static utility", cluster_scope=reviewed_scope)
        for card in cards
    ])
    build_landscape(run, review)

    tampered_scope = {"object": "other unit", "time_horizon": "other horizon", "setting": "other setting"}
    tampered_key = canonical_scope_key(tampered_scope)
    edge_path = run / "landscape" / "card_concept_edges.csv"
    with edge_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        edges = list(reader)
    for edge in edges:
        if edge["normalized_text"] == "static utility":
            edge["cluster_scope"] = json.dumps(tampered_scope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            edge["cluster_scope_key"] = tampered_key
    with edge_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(edges)
    assumptions_path = run / "landscape" / "assumptions.json"
    assumptions = json.loads(assumptions_path.read_text(encoding="utf-8"))
    entry = next(row for row in assumptions["entries"] if row["normalized_text"] == "static utility")
    entry["scope"] = tampered_scope
    entry["scope_key"] = tampered_key
    assumptions_path.write_text(json.dumps(assumptions, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="assignment|replay|projection"):
        validate_landscape_bundle(run)


@pytest.mark.parametrize("kind", ["duplicate", "bad", "stale"])
def test_review_override_artifact_rejects_duplicate_bad_and_stale_rows(tmp_path: Path, kind: str) -> None:
    from idea_factory.landscape import build_landscape, canonical_scope_key

    run, jobs = _bundle(tmp_path)
    card = _card(jobs[0], "one")
    ingest_card_results(run / "cards" / "card_jobs.jsonl", [_result(jobs[0], [card]), _result(jobs[1], [])])
    row = _override("one", "cache stays warm", canonical_scope_key(card["scope"]), normalized="warm cache")
    rows = [row, row.copy()] if kind == "duplicate" else [row]
    if kind == "bad":
        rows[0]["merge_reason"] = " "
    if kind == "stale":
        rows[0]["card_id"] = "gone"
    review = _review_artifact(tmp_path / "review.json", rows)
    with pytest.raises(ValueError, match="duplicate|approved override|stale or orphan"):
        build_landscape(run, review)
    assert not any((run / "landscape").glob("*"))


def test_failure_statuses_unknown_absence_and_delimiter_safe_scopes(tmp_path: Path) -> None:
    from idea_factory.landscape import build_landscape

    run, jobs = _bundle(tmp_path)
    first = _card(jobs[0], "one", failure_mechanism={"text": "", "status": "UNKNOWN"}, scope={"object": "a|b", "time_horizon": "c", "setting": "d"})
    second = _card(jobs[1], "two", failure_mechanism={"text": "shift", "status": "INFERRED"}, scope={"object": "a", "time_horizon": "b|c", "setting": "d"})
    ingest_card_results(run / "cards" / "card_jobs.jsonl", [_result(jobs[1], [second]), _result(jobs[0], [first])])
    build_landscape(run)
    failures = json.loads((run / "landscape" / "failures.json").read_text(encoding="utf-8"))
    assert failures["missing_unknown_failure_mechanism_count"] == 1
    mechanism_entries = [row for row in failures["entries"] if row["facet"] == "failure_mechanism"]
    assert len(mechanism_entries) == 1 and mechanism_entries[0]["evidence_statuses"] == ["INFERRED"]
    observed = [row for row in failures["entries"] if row["facet"] == "failure_observation"]
    assert len(observed) == 2
    assert observed[0]["scope_key"] != observed[1]["scope_key"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "orphan", "stale"])
def test_landscape_refuses_partial_or_tampered_task5_bundle(tmp_path: Path, mutation: str) -> None:
    from idea_factory.landscape import build_landscape

    run, jobs = _bundle(tmp_path)
    ingest_card_results(run / "cards" / "card_jobs.jsonl", [_result(jobs[0], [_card(jobs[0], "one")]), _result(jobs[1], [])])
    accepted = run / "results" / "paper_cards.jsonl"
    if mutation == "missing":
        (run / "results" / "card_job_outcomes.jsonl").unlink()
    elif mutation == "duplicate":
        accepted.write_text(accepted.read_text(encoding="utf-8") * 2, encoding="utf-8")
    elif mutation == "orphan":
        row = json.loads(accepted.read_text(encoding="utf-8"))
        row["job_id"] = "rogue"
        accepted.write_text(json.dumps(row) + "\n", encoding="utf-8")
    else:
        selection = run / "corpus" / "selection_manifest.jsonl"
        selection.write_text(selection.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        build_landscape(run)


def test_landscape_requires_rejected_rows_to_exactly_cover_partial_outcome(tmp_path: Path) -> None:
    from idea_factory.landscape import build_landscape

    run, jobs = _bundle(tmp_path)
    bad = _card(jobs[0], "bad")
    bad["evidence_pointers"] = []
    ingest_card_results(run / "cards" / "card_jobs.jsonl", [_result(jobs[0], [_card(jobs[0], "good"), bad]), _result(jobs[1], [])])
    rejected = run / "results" / "rejected_paper_cards.jsonl"
    rejected.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="rejected"):
        build_landscape(run)


def test_landscape_rehashes_a_fully_accepted_result_to_detect_semantic_tampering(tmp_path: Path) -> None:
    from idea_factory.landscape import build_landscape

    run, jobs = _bundle(tmp_path)
    ingest_card_results(run / "cards" / "card_jobs.jsonl", [_result(jobs[0], [_card(jobs[0], "one")]), _result(jobs[1], [])])
    accepted = run / "results" / "paper_cards.jsonl"
    row = json.loads(accepted.read_text(encoding="utf-8"))
    row["card"]["assumption"]["text"] = "a different but still nonblank claim"
    accepted.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="accepted cards projection hash|raw result hash"):
        build_landscape(run)


@pytest.mark.parametrize("mutation", ["assumption", "card", "provenance"])
def test_partial_projection_detects_accepted_tampering(tmp_path: Path, mutation: str) -> None:
    from idea_factory.landscape import build_landscape

    run, jobs = _bundle(tmp_path)
    bad = _card(jobs[0], "bad")
    bad["evidence_pointers"] = []
    ingest_card_results(
        run / "cards" / "card_jobs.jsonl",
        [_result(jobs[0], [_card(jobs[0], "good"), bad]), _result(jobs[1], [])],
    )
    accepted = run / "results" / "paper_cards.jsonl"
    row = json.loads(accepted.read_text(encoding="utf-8"))
    if mutation == "assumption":
        row["card"]["assumption"]["text"] = "tampered but coherent"
    elif mutation == "card":
        row["card"]["paper"]["title"] = "tampered title"
    else:
        row["raw_result_sha256"] = "f" * 64
    accepted.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="projection|binding"):
        build_landscape(run)


@pytest.mark.parametrize("mutation", ["error", "index", "card_id"])
def test_partial_projection_detects_rejected_metadata_tampering(tmp_path: Path, mutation: str) -> None:
    from idea_factory.landscape import build_landscape

    run, jobs = _bundle(tmp_path)
    bad = _card(jobs[0], "bad")
    bad["evidence_pointers"] = []
    ingest_card_results(
        run / "cards" / "card_jobs.jsonl",
        [_result(jobs[0], [_card(jobs[0], "good"), bad]), _result(jobs[1], [])],
    )
    rejected = run / "results" / "rejected_paper_cards.jsonl"
    row = json.loads(rejected.read_text(encoding="utf-8"))
    if mutation == "error":
        row["error"] = "changed rejection reason"
    elif mutation == "index":
        row["index"] = 99
    else:
        row["card_id"] = "other"
    rejected.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rejected.*projection|outcome card IDs"):
        build_landscape(run)


@pytest.mark.parametrize("projection", ["accepted", "rejected"])
def test_partial_projection_order_is_bound(tmp_path: Path, projection: str) -> None:
    from idea_factory.landscape import build_landscape

    run, jobs = _bundle(tmp_path)
    bad_one = _card(jobs[0], "bad-one")
    bad_one["evidence_pointers"] = []
    bad_two = _card(jobs[0], "bad-two")
    bad_two["evidence_pointers"] = []
    ingest_card_results(
        run / "cards" / "card_jobs.jsonl",
        [_result(jobs[0], [_card(jobs[0], "good-one"), _card(jobs[0], "good-two"), bad_one, bad_two]), _result(jobs[1], [])],
    )
    target = run / "results" / "card_job_outcomes.jsonl"
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    field = "accepted_card_ids" if projection == "accepted" else "rejected_card_ids"
    rows[0][field] = list(reversed(rows[0][field]))
    target.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="projection|accepted cards|outcome card IDs"):
        build_landscape(run)


def test_build_is_byte_deterministic_and_publish_failure_removes_all_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import idea_factory.landscape as landscape_module
    from idea_factory.landscape import build_landscape

    run, jobs = _bundle(tmp_path)
    ingest_card_results(run / "cards" / "card_jobs.jsonl", [_result(jobs[1], [_card(jobs[1], "two")]), _result(jobs[0], [_card(jobs[0], "one")])])
    build_landscape(run)
    original = {path.name: path.read_bytes() for path in (run / "landscape").iterdir()}
    build_landscape(run)
    assert original == {path.name: path.read_bytes() for path in (run / "landscape").iterdir()}

    real_replace = landscape_module.os.replace
    calls = 0
    def fail_second(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("publish failure")
        return real_replace(source, target)
    monkeypatch.setattr(landscape_module.os, "replace", fail_second)
    with pytest.raises(OSError, match="publish failure"):
        build_landscape(run)
    assert not any((run / "landscape").glob("*"))


def test_review_override_order_does_not_change_landscape_bytes(tmp_path: Path) -> None:
    from idea_factory.landscape import build_landscape, canonical_scope_key

    run, jobs = _bundle(tmp_path)
    first = _card(jobs[0], "one", assumption={"text": "importance score", "status": "REPORTED"})
    second = _card(jobs[1], "two", assumption={"text": "significance score", "status": "REPORTED"})
    ingest_card_results(run / "cards" / "card_jobs.jsonl", [_result(jobs[0], [first]), _result(jobs[1], [second])])
    scope = canonical_scope_key(first["scope"])
    alpha = _override("one", "importance score", scope, normalized="score")
    beta = _override("two", "significance score", scope, normalized="score")
    first_review = _review_artifact(tmp_path / "a.json", [alpha, beta])
    second_review = _review_artifact(tmp_path / "b.json", [beta, alpha])
    build_landscape(run, first_review)
    first_bytes = {path.name: path.read_bytes() for path in (run / "landscape").iterdir()}
    build_landscape(run, second_review)
    assert first_bytes == {path.name: path.read_bytes() for path in (run / "landscape").iterdir()}


@pytest.mark.parametrize(
    ("field", "value"),
    [("reviewed", "true"), ("disputed", "false"), ("dimension", " assumptions ")],
)
def test_reviewed_assignment_rejects_coerced_or_trimmed_json_values(tmp_path: Path, field: str, value: object) -> None:
    from idea_factory.landscape import build_landscape, canonical_scope_key

    run, jobs = _bundle(tmp_path)
    card = _card(jobs[0], "one")
    ingest_card_results(run / "cards" / "card_jobs.jsonl", [_result(jobs[0], [card]), _result(jobs[1], [])])
    row = _override("one", "cache stays warm", canonical_scope_key(card["scope"]), normalized="warm cache")
    row[field] = value
    review = _review_artifact(tmp_path / "coerced-review.json", [row])
    with pytest.raises(ValueError, match="strict|normalizes|invalid reviewed assignment"):
        build_landscape(run, review)


def test_reviewed_assignment_ignores_json_object_key_order(tmp_path: Path) -> None:
    from idea_factory.landscape import build_landscape, canonical_scope_key

    run, jobs = _bundle(tmp_path)
    card = _card(jobs[0], "one")
    ingest_card_results(run / "cards" / "card_jobs.jsonl", [_result(jobs[0], [card]), _result(jobs[1], [])])
    row = _override("one", "cache stays warm", canonical_scope_key(card["scope"]), normalized="warm cache")
    reversed_row = dict(reversed(tuple(row.items())))
    build_landscape(run, _review_artifact(tmp_path / "reordered-review.json", [reversed_row]))
    assumptions = json.loads((run / "landscape" / "assumptions.json").read_text(encoding="utf-8"))
    assert assumptions["entries"][0]["normalized_text"] == "warm cache"


def test_landscape_junction_escape_does_not_touch_external_files(tmp_path: Path) -> None:
    from idea_factory.landscape import build_landscape

    run, jobs = _bundle(tmp_path)
    ingest_card_results(run / "cards" / "card_jobs.jsonl", [_result(jobs[0], []), _result(jobs[1], [])])
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "assumptions.json"
    sentinel.write_text("sentinel", encoding="utf-8")
    link = run / "landscape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        completed = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)], capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            pytest.skip("directory links unavailable")
    with pytest.raises(ValueError, match="escapes"):
        build_landscape(run)
    assert sentinel.read_text(encoding="utf-8") == "sentinel"


@pytest.mark.parametrize("linked_name", ["cards", "results", "corpus"])
def test_landscape_rejects_upstream_directory_link_to_another_run(tmp_path: Path, linked_name: str) -> None:
    from idea_factory.landscape import build_landscape

    run, jobs = _bundle(tmp_path)
    ingest_card_results(run / "cards" / "card_jobs.jsonl", [_result(jobs[0], [_card(jobs[0])]), _result(jobs[1], [])])
    outside = tmp_path / "outside" / linked_name
    outside.parent.mkdir()
    (run / linked_name).rename(outside)
    sentinel = outside / "outside-sentinel.txt"
    sentinel.write_text("untouched", encoding="utf-8")
    link = run / linked_name
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        completed = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)], capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            pytest.skip("directory links unavailable")
    with pytest.raises(ValueError, match="expected run|anchored|same active run"):
        build_landscape(run)
    assert sentinel.read_text(encoding="utf-8") == "untouched"
    assert not any((run / "landscape").glob("*"))
