from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")


class FixtureBackend:
    """Offline stage double: the strict stage APIs have their own test suites."""

    def emit_corpus_jobs(self, run: Path, **_: object) -> None:
        _jsonl(run / "corpus" / "router_jobs.jsonl", [{"job_id": "router-1"}])

    def ingest_corpus_labels(self, run: Path, **_: object) -> None:
        _jsonl(run / "corpus" / "selection_manifest.jsonl", [{"record_id": "paper-1"}])
        _jsonl(run / "corpus" / "rejected_manifest.jsonl", [])

    def emit_card_jobs(self, run: Path, **_: object) -> None:
        _jsonl(run / "cards" / "card_jobs.jsonl", [{"job_id": "card-job-1"}])

    def ingest_cards(self, run: Path, **_: object) -> None:
        _jsonl(run / "results" / "paper_cards.jsonl", [{"record_id": "card-1"}])
        _jsonl(run / "results" / "card_job_outcomes.jsonl", [{"status": "ACCEPTED"}])

    def build_landscape(self, run: Path, **_: object) -> None:
        directory = run / "landscape"
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("assumptions.json", "failures.json", "mechanisms.json", "evaluations.json"):
            (directory / name).write_text("{}\n", encoding="utf-8")
        (directory / "card_concept_edges.csv").write_text("card_id\n", encoding="utf-8")

    def emit_opportunity_jobs(self, run: Path, **_: object) -> None:
        _jsonl(run / "opportunities" / "mining_jobs.jsonl", [{"job_id": "opp-job-1"}])
        _jsonl(run / "opportunities" / "cluster_audit.jsonl", [])

    def ingest_opportunities(self, run: Path, **_: object) -> None:
        _jsonl(run / "opportunities" / "opportunity_candidates.jsonl", [{"record_id": "opp-1"}])
        _jsonl(run / "opportunities" / "rejected_results.jsonl", [])
        _jsonl(run / "opportunities" / "result_outcomes.jsonl", [{"status": "ACCEPTED"}])

    def quality_gate(self, run: Path, **_: object) -> None:
        _jsonl(run / "quality" / "ready_for_internal_dedup.jsonl", [{"record_id": "opp-1"}])
        _jsonl(run / "quality" / "rejected.jsonl", [])
        _jsonl(run / "quality" / "opportunity_job_outcomes.jsonl", [{"status": "READY"}])

    def internal_dedup(self, run: Path, **_: object) -> None:
        _jsonl(run / "ledger" / "ready_after_internal_dedup.jsonl", [{"record_id": "opp-1"}])
        _jsonl(run / "ledger" / "dedup_outcomes.jsonl", [{"status": "CLEAR"}])

    def emit_recon_pack(self, run: Path, **_: object) -> None:
        _jsonl(run / "recon" / "query_pack.jsonl", [{"query_id": "q-1"}])
        (run / "recon" / "query_pack_manifest.json").write_text("{}", encoding="utf-8")
        _jsonl(run / "recon" / "execution_job_templates.jsonl", [])
        _jsonl(run / "recon" / "execution_jobs.jsonl", [])

    def ingest_recon(self, run: Path, **_: object) -> None:
        _jsonl(run / "recon" / "reports.jsonl", [{"record_id": "recon-1"}])
        _jsonl(run / "recon" / "outcomes.jsonl", [{"status": "ACCEPTED"}])
        _jsonl(run / "recon" / "ready_for_routes.jsonl", [{"record_id": "opp-1", "live_ready": False}])

    def emit_route_jobs(self, run: Path, **_: object) -> None:
        _jsonl(run / "routes" / "jobs.jsonl", [{"job_id": "route-job-1"}])

    def ingest_routes(self, run: Path, **_: object) -> None:
        _jsonl(run / "routes" / "results.jsonl", [{"record_id": "route-1"}])
        _jsonl(run / "routes" / "outcomes.jsonl", [{"status": "ACCEPTED"}])
        (run / "routes" / "bundle_manifest.json").write_text("{}", encoding="utf-8")

    def emit_review_jobs(self, run: Path, **_: object) -> None:
        _jsonl(run / "review" / "jobs.jsonl", [{"job_id": "review-job-1"}])

    def ingest_reviews(self, run: Path, **_: object) -> None:
        _jsonl(run / "review" / "results.jsonl", [{"record_id": "review-1", "decision": "PASS_TO_HUMAN"}])
        _jsonl(run / "review" / "outcomes.jsonl", [{"job_id": "review-job-1", "status": "ACCEPTED"}])
        (run / "review" / "bundle_manifest.json").write_text("{}", encoding="utf-8")

    def ingest_human_scores(self, run: Path, **_: object) -> None:
        _jsonl(run / "ledger" / "human_scores.jsonl", [{"record_id": "human-1"}])
        _jsonl(run / "ledger" / "human_scoring_outcomes.jsonl", [{"status": "ACCEPTED"}])
        (run / "ledger" / "human_bundle_manifest.json").write_text("{}", encoding="utf-8")

    def finalize(self, run: Path, **_: object) -> dict[str, object]:
        from idea_factory.models import DecisiveTest, HumanScores, IdeaPack, IdeaStatus, NearestPrior

        packs = run / "idea_packs"
        packs.mkdir(parents=True, exist_ok=True)
        model = IdeaPack(
            idea_id="idea-1", status=IdeaStatus.READY_FOR_CHEAP_TEST,
            opportunity="bound opportunity", core_hypothesis="bound hypothesis", why_now="now",
            inference_flags=("fixture inference",),
            nearest_priors=(NearestPrior(paper="prior", exact_overlap="overlap", residual_difference="difference", evidence_url_or_id="fixture:prior"),),
            proposed_mechanism="mechanism", source_of_gain="isolated gain",
            cheapest_decisive_test=DecisiveTest(setup="fixture setup", discriminates_against="alternative", expected_runtime_or_cost="offline"),
            strongest_baseline="matched baseline", kill_condition="no measured gain",
            main_uncertainty="uncertainty", expected_reviewer_2_objection="just A+B",
            response_to_objection="isolated source", evidence_that_would_make_reviewer_correct="no isolation",
            human_scores=HumanScores(specific_novelty=4, importance=4, paper_potential=4, feasibility=4, excitement=4, evidence_clarity=4),
            human_reason_codes=("WANT_TO_TEST_NOW",),
        )
        (packs / "idea-1.json").write_text(json.dumps({"idea_id": "idea-1", "status": "READY_FOR_CHEAP_TEST"}), encoding="utf-8")
        (packs / "manifest.json").write_text(json.dumps({"pack_count": 1, "ready_count": 1}), encoding="utf-8")
        return {"pack_count": 1, "ready_count": 1, "ready_packs": [model]}


def _invoke(*args: str) -> int:
    from idea_factory.cli import main

    return main(list(args))


def _config(tmp_path: Path) -> Path:
    notes = tmp_path / "notes"; notes.mkdir()
    note = notes / "paper.md"; note.write_text("fixture note", encoding="utf-8")
    papers = tmp_path / "papers"; papers.mkdir()
    candidates = papers / "list.txt"; candidates.write_text(str(note.resolve()) + "\n", encoding="utf-8")
    legacy = papers / "idea_ledger.md"; legacy.write_text("legacy", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "notes_root": "notes", "candidate_lists": ["papers/list.txt"],
        "legacy_ledger": "papers/idea_ledger.md", "target_min": 1, "target_max": 1,
        "allowed_labels": ["KV_CACHE"], "bridge_regression_slugs": [],
    }), encoding="utf-8")
    return config


def _write_manual_quality_bundle(run: Path) -> None:
    _jsonl(run / "quality" / "ready_for_internal_dedup.jsonl", [])
    _jsonl(run / "quality" / "rejected.jsonl", [])
    _jsonl(run / "quality" / "opportunity_job_outcomes.jsonl", [])


def _write_manual_human_bundle(run: Path) -> None:
    _jsonl(run / "ledger" / "human_scores.jsonl", [])
    _jsonl(run / "ledger" / "human_scoring_outcomes.jsonl", [])
    (run / "ledger" / "human_bundle_manifest.json").write_text("{}\n", encoding="utf-8")


def _write_manual_recon_bundle(run: Path) -> None:
    _jsonl(run / "recon" / "reports.jsonl", [])
    _jsonl(run / "recon" / "outcomes.jsonl", [])
    _jsonl(run / "recon" / "ready_for_routes.jsonl", [])


def _write_manual_route_bundle(run: Path) -> None:
    _jsonl(run / "routes" / "results.jsonl", [])
    _jsonl(run / "routes" / "outcomes.jsonl", [])
    (run / "routes" / "bundle_manifest.json").write_text("{}\n", encoding="utf-8")


def _real_human_scored_pipeline_run(
    tmp_path: Path, *, route_prompt: Path | None = None, review_prompt: Path | None = None,
):
    from idea_factory.artifacts import read_jsonl, sha256_file
    from idea_factory.ledger import emit_human_scoring_jobs, ingest_human_scores
    from idea_factory.models import RunStage, RunState
    from idea_factory.review import emit_review_jobs, ingest_review_results
    from idea_factory.routes import emit_route_jobs, ingest_route_results
    from test_ledger import _human_score, _task11_reviewed_run
    from test_routes_review import _ready_run, _review_result, _route_result

    if route_prompt is None and review_prompt is None:
        run, resolved = _task11_reviewed_run(
            tmp_path, ("PASS_TO_HUMAN", "KILL", "KILL"), near_prior=True,
        )
    else:
        assert route_prompt is not None and review_prompt is not None
        run, resolved = _ready_run(tmp_path, decision="NEAR_PRIOR_WITH_RESIDUAL")
        route_jobs = emit_route_jobs(
            run, resolved, prompt_path=route_prompt, allow_test_ready=True,
        )
        route_job = read_jsonl(route_jobs)[0]
        ingest_route_results(
            route_jobs, [_route_result(route_job)], resolved,
            prompt_path=route_prompt, allow_test_ready=True,
        )
        review_jobs = emit_review_jobs(
            run, resolved, prompt_path=review_prompt, route_prompt_path=route_prompt,
            allow_test_ready=True,
        )
        review_rows = read_jsonl(review_jobs)
        decisions = ("PASS_TO_HUMAN", "KILL", "KILL")
        ingest_review_results(
            review_jobs,
            [_review_result(job, verdict) for job, verdict in zip(review_rows, decisions)],
            resolved, prompt_path=review_prompt, route_prompt_path=route_prompt,
            allow_test_ready=True,
        )
    human_jobs = emit_human_scoring_jobs(
        run, resolved, prompt_path=review_prompt, route_prompt_path=route_prompt,
        allow_test_ready=True,
    )
    human_job = read_jsonl(human_jobs)[0]
    ingest_human_scores(
        human_jobs, [_human_score(human_job)], resolved,
        prompt_path=review_prompt, route_prompt_path=route_prompt, allow_test_ready=True,
    )
    config = tmp_path / "pipeline-config.json"
    config.write_text(json.dumps({
        "notes_root": str(resolved.notes_root),
        "candidate_lists": [str(path) for path in resolved.candidate_lists],
        "legacy_ledger": str(resolved.legacy_ledger),
        "target_min": resolved.target_min,
        "target_max": resolved.target_max,
        "allowed_labels": list(resolved.allowed_labels),
        "bridge_regression_slugs": list(resolved.bridge_regression_slugs),
    }), encoding="utf-8")
    run_id = f"pipeline-{tmp_path.name}"
    (run / "manifest.json").write_text(json.dumps({
        "schema_version": "idea_factory.run_manifest.v1",
        "run_id": run_id,
        "mode": "offline-fixture",
        "config_path": str(config.resolve()),
        "config_sha256": sha256_file(config),
        "config_repo_root": str(tmp_path.resolve()),
        "created_at": "2026-08-04T12:00:00+08:00",
        "truth_boundary": "OFFLINE_FIXTURE_NOT_LIVE_RECON",
        "cheap_test_executed": False,
        "subsequent_result": None,
    }), encoding="utf-8")
    state = RunState(run_id=run_id, stage=RunStage.HUMAN_SCORED)
    (run / "state.json").write_text(json.dumps(state.model_dump(mode="json")), encoding="utf-8")
    return run, config, resolved


def _seed_persistent_ledger(tmp_path: Path, repository: Path) -> bytes:
    from idea_factory.artifacts import read_jsonl
    from idea_factory.ledger import append_ledger_updates, emit_human_scoring_jobs, ingest_human_scores
    from idea_factory.render import render_idea_packs
    from test_ledger import _human_score, _task11_reviewed_run

    prior = tmp_path / "prior"; prior.mkdir()
    run, config = _task11_reviewed_run(
        prior, ("PASS_TO_HUMAN", "KILL", "KILL"), near_prior=True,
    )
    jobs = emit_human_scoring_jobs(run, config, allow_test_ready=True)
    job = read_jsonl(jobs)[0]
    ingest_human_scores(jobs, [_human_score(job)], config, allow_test_ready=True)
    render_idea_packs(run, config, allow_test_ready=True)
    append_ledger_updates(run, config, repository_root=repository, allow_test_ready=True)
    return (repository / "data" / "idea_ledger.jsonl").read_bytes()


def _real_zero_survivor_pipeline_run(tmp_path: Path, *, allow_test_ready: bool):
    from idea_factory.artifacts import read_jsonl, sha256_file
    from idea_factory.ledger import emit_human_scoring_jobs, ingest_human_scores
    from idea_factory.models import RunStage, RunState
    from idea_factory.review import emit_review_jobs, ingest_review_results
    from idea_factory.routes import emit_route_jobs, ingest_route_results
    from test_routes_review import _zero_recon_run

    run, resolved = _zero_recon_run(tmp_path)
    route_jobs = emit_route_jobs(run, resolved, allow_test_ready=allow_test_ready)
    assert read_jsonl(route_jobs) == []
    ingest_route_results(route_jobs, [], resolved, allow_test_ready=allow_test_ready)
    review_jobs = emit_review_jobs(run, resolved, allow_test_ready=allow_test_ready)
    assert read_jsonl(review_jobs) == []
    ingest_review_results(review_jobs, [], resolved, allow_test_ready=allow_test_ready)
    human_jobs = emit_human_scoring_jobs(run, resolved, allow_test_ready=allow_test_ready)
    assert read_jsonl(human_jobs) == []
    ingest_human_scores(human_jobs, [], resolved, allow_test_ready=allow_test_ready)
    config = tmp_path / "pipeline-config.json"
    config.write_text(json.dumps({
        "notes_root": str(resolved.notes_root),
        "candidate_lists": [str(path) for path in resolved.candidate_lists],
        "legacy_ledger": str(resolved.legacy_ledger),
        "target_min": resolved.target_min,
        "target_max": resolved.target_max,
        "allowed_labels": list(resolved.allowed_labels),
        "bridge_regression_slugs": list(resolved.bridge_regression_slugs),
    }), encoding="utf-8")
    run_id = f"zero-pipeline-{tmp_path.name}-{allow_test_ready}"
    (run / "manifest.json").write_text(json.dumps({
        "schema_version": "idea_factory.run_manifest.v1", "run_id": run_id,
        "mode": "offline-fixture", "config_path": str(config.resolve()),
        "config_sha256": sha256_file(config), "config_repo_root": str(tmp_path.resolve()),
        "created_at": "2026-08-04T12:00:00+08:00",
        "truth_boundary": "OFFLINE_FIXTURE_NOT_LIVE_RECON",
        "cheap_test_executed": False, "subsequent_result": None,
    }), encoding="utf-8")
    state = RunState(run_id=run_id, stage=RunStage.REVIEWED)
    (run / "state.json").write_text(json.dumps(state.model_dump(mode="json")), encoding="utf-8")
    return run, config, resolved


def test_offline_cli_end_to_end_emits_required_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from idea_factory import pipeline

    monkeypatch.setattr(pipeline, "BACKEND", FixtureBackend())
    run = tmp_path / "run"
    config = _config(tmp_path)
    fixture_files = {}
    for name in ("labels", "cards", "opportunities", "recon", "routes", "reviews", "human"):
        path = tmp_path / f"{name}.jsonl"
        _jsonl(path, [{"fixture": name}])
        fixture_files[name] = path

    assert _invoke("init-run", "--run", str(run), "--config", str(config), "--mode", "offline-fixture") == 0
    commands = [
        ("emit-corpus-jobs", []),
        ("ingest-corpus-labels", ["--results", str(fixture_files["labels"])]),
        ("emit-card-jobs", []),
        ("ingest-cards", ["--results", str(fixture_files["cards"])]),
        ("build-landscape", []),
        ("emit-opportunity-jobs", []),
        ("ingest-opportunities", ["--results", str(fixture_files["opportunities"])]),
        ("quality-gate", []),
        ("internal-dedup", ["--results", str(fixture_files["opportunities"])]),
        ("emit-recon-pack", []),
        ("ingest-recon", ["--results", str(fixture_files["recon"]), "--allow-test-ready"]),
        ("emit-route-jobs", ["--allow-test-ready"]),
        ("ingest-routes", ["--results", str(fixture_files["routes"]), "--allow-test-ready"]),
        ("emit-review-jobs", ["--allow-test-ready"]),
        ("ingest-reviews", ["--results", str(fixture_files["reviews"]), "--allow-test-ready"]),
        ("ingest-human-scores", ["--results", str(fixture_files["human"]), "--allow-test-ready"]),
        ("finalize", ["--allow-test-ready"]),
    ]
    for command, extra in commands:
        assert _invoke(command, "--run", str(run), "--config", str(config), *extra) == 0

    required = [
        "manifest.json", "state.json", "corpus/selection_manifest.jsonl",
            "results/paper_cards.jsonl", "landscape/assumptions.json",
        "quality/ready_for_internal_dedup.jsonl", "recon/query_pack.jsonl",
        "recon/reports.jsonl", "routes/results.jsonl", "review/results.jsonl",
        "idea_packs/idea-1.json", "report.md",
    ]
    assert all((run / relpath).is_file() for relpath in required)

    assert _invoke("status", "--run", str(run)) == 0
    status = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert status["stage"] == "COMPLETE"
    assert status["counts"] == {
        "completed": 1, "awaiting-external-result": 0, "killed": 0,
        "surviving": 0, "unresolved-error": 1,
    }
    assert status["offline_tests_ready"] is True
    assert status["live_recon_executed"] is False
    assert status["human_review_completed"] is True
    assert status["idea_pack_produced"] is False
    assert status["real_idea_pack_produced"] is False
    assert status["cheap_test_executed"] is False


def test_stage_order_idempotency_and_changed_inputs_require_new_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from idea_factory import pipeline
    from idea_factory.cli import main

    monkeypatch.setattr(pipeline, "BACKEND", FixtureBackend())
    run = tmp_path / "run"; config = _config(tmp_path)
    results = tmp_path / "labels.jsonl"; _jsonl(results, [{"fixture": "v1"}])
    assert main(["init-run", "--run", str(run), "--config", str(config)]) == 0
    with pytest.raises(SystemExit):
        main(["ingest-cards", "--run", str(run), "--config", str(config), "--results", str(results)])
    assert main(["emit-corpus-jobs", "--run", str(run), "--config", str(config)]) == 0
    assert main(["ingest-corpus-labels", "--run", str(run), "--config", str(config), "--results", str(results)]) == 0
    before = (run / "state.json").read_bytes()
    assert main(["ingest-corpus-labels", "--run", str(run), "--config", str(config), "--results", str(results)]) == 0
    assert (run / "state.json").read_bytes() == before
    _jsonl(results, [{"fixture": "v2"}])
    with pytest.raises(SystemExit):
        main(["ingest-corpus-labels", "--run", str(run), "--config", str(config), "--results", str(results)])
    assert "--new-run" in capsys.readouterr().err


def test_all_explicit_subcommands_are_registered() -> None:
    from idea_factory.cli import build_parser

    parser = build_parser()
    action = next(action for action in parser._actions if action.dest == "command")
    assert set(action.choices) == {
        "init-run", "emit-corpus-jobs", "ingest-corpus-labels", "emit-card-jobs",
        "ingest-cards", "build-landscape", "emit-opportunity-jobs",
        "ingest-opportunities", "quality-gate", "internal-dedup", "emit-recon-pack",
        "ingest-recon", "emit-route-jobs", "ingest-routes", "emit-review-jobs",
        "ingest-reviews", "ingest-human-scores", "finalize", "status",
    }


def test_real_backend_routes_labels_cards_and_landscape_offline(tmp_path: Path) -> None:
    from idea_factory.artifacts import read_jsonl
    from idea_factory.cli import main

    run = tmp_path / "run"; config = _config(tmp_path)
    assert main(["init-run", "--run", str(run), "--config", str(config), "--mode", "offline-fixture"]) == 0
    assert main(["emit-corpus-jobs", "--run", str(run), "--config", str(config)]) == 0
    router_job = read_jsonl(run / "corpus" / "router_jobs.jsonl")[0]
    labels = tmp_path / "labels.jsonl"
    _jsonl(labels, [{
        "schema_version": "idea_factory.corpus_router_result.v2",
        **{key: router_job[key] for key in ("job_id", "slug", "note_sha256", "prompt_sha256")},
        "label": "KV_CACHE", "core_mechanism": "bounded eviction",
        "scope_reason": "KV serving", "evidence_locator": "note:1", "confidence": "HIGH",
    }])
    assert main(["ingest-corpus-labels", "--run", str(run), "--config", str(config), "--results", str(labels)]) == 0
    assert main(["emit-card-jobs", "--run", str(run), "--config", str(config)]) == 0
    card_job = read_jsonl(run / "cards" / "card_jobs.jsonl")[0]
    cards = tmp_path / "cards.jsonl"
    card = {
        "schema_version": "idea_factory.paper_card.v1", "card_id": "card-pipeline",
        "paper": {"title": "Fixture", "year": 2026, "venue": "Test", "source_path": card_job["note_path"]},
        "problem": "retrieval latency", "assumption": {"text": "cache stays resident", "status": "REPORTED"},
        "mechanism": "eviction", "failure_observation": {"text": "latency rises", "status": "OBSERVED"},
        "failure_mechanism": {"text": "working set shifts", "status": "INFERRED"},
        "limitation": "one trace", "evaluation": {"measurement": "latency", "regime": "long context"},
        "scope": {"object": "cache", "time_horizon": "hours", "setting": "serving"},
        "evidence_pointers": [{
            "source": "NOTE", "locator": card_job["note_path"] + "# Evidence",
            "supports": "problem,assumption,mechanism,failure_observation,failure_mechanism,limitation,evaluation.measurement,evaluation.regime,scope.object,scope.time_horizon,scope.setting",
        }],
        "extraction_confidence": "HIGH",
    }
    _jsonl(cards, [{
        "schema_version": "idea_factory.paper_card_result.v1",
        **{key: card_job[key] for key in ("job_id", "slug", "note_sha256", "prompt_sha256")},
        "cards": [card],
    }])
    assert main(["ingest-cards", "--run", str(run), "--config", str(config), "--results", str(cards)]) == 0
    assert main(["build-landscape", "--run", str(run), "--config", str(config)]) == 0
    assert (run / "results" / "paper_cards.jsonl").is_file()
    assert list((run / "landscape").glob("*.json"))


def test_finalize_zero_survivor_is_operational_completion_not_idea_yield(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from idea_factory import pipeline
    from idea_factory.models import RunStage, RunState

    class ZeroBackend(FixtureBackend):
        def finalize(self, run: Path, **_: object) -> dict[str, object]:
            directory = run / "idea_packs"; directory.mkdir(parents=True, exist_ok=True)
            (directory / "manifest.json").write_text(json.dumps({"pack_count": 0, "ready_count": 0}), encoding="utf-8")
            return {"pack_count": 0, "ready_count": 0, "ready_packs": []}

    monkeypatch.setattr(pipeline, "BACKEND", ZeroBackend())
    run = tmp_path / "run"; config = _config(tmp_path)
    pipeline.init_run(run, config, mode="offline-fixture")
    state = RunState.model_validate(json.loads((run / "state.json").read_text(encoding="utf-8")))
    while state.stage != RunStage.REVIEWED:
        state.advance(RunStage(list(RunStage)[list(RunStage).index(state.stage) + 1]))
    (run / "state.json").write_text(json.dumps(state.model_dump(mode="json")), encoding="utf-8")
    _write_manual_human_bundle(run)
    assert pipeline.finalize(run, config, allow_test_ready=True) == "COMPLETED"
    result = pipeline.status(run)
    assert result["completion_kind"] == "AUDITABLE_ZERO_SURVIVOR"
    assert result["operational_completion"] is True
    assert result["idea_pack_produced"] is False
    assert "Cheap test executed: false" in (run / "report.md").read_text(encoding="utf-8")


def test_external_handoff_is_awaiting_not_error_or_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from idea_factory import pipeline
    from idea_factory.models import RunStage, RunState

    class AwaitingBackend(FixtureBackend):
        def internal_dedup(self, run: Path, **_: object) -> bool:
            _jsonl(run / "ledger" / "shape_assignment_jobs.jsonl", [{"job_id": "shape-1"}])
            return False

    monkeypatch.setattr(pipeline, "BACKEND", AwaitingBackend())
    run = tmp_path / "run"; config = _config(tmp_path)
    pipeline.init_run(run, config, mode="offline-fixture")
    state = RunState.model_validate(json.loads((run / "state.json").read_text(encoding="utf-8")))
    for target in (RunStage.CORPUS_ROUTED, RunStage.CARDS_READY, RunStage.LANDSCAPE_READY, RunStage.OPPORTUNITIES_READY, RunStage.QUALITY_GATED):
        state.advance(target)
    (run / "state.json").write_text(json.dumps(state.model_dump(mode="json")), encoding="utf-8")
    _write_manual_quality_bundle(run)
    fixture = tmp_path / "shape.jsonl"; _jsonl(fixture, [{"fixture": "shape"}])
    assert pipeline.execute("internal-dedup", run, config) == "AWAITING_EXTERNAL_RESULT"
    current = pipeline.status(run)["counts"]
    assert current["awaiting-external-result"] == 1
    assert current["unresolved-error"] == 0
    assert current["completed"] == 0


def test_run_lock_prevents_concurrent_stage_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from idea_factory import pipeline
    from idea_factory.stage_io import stage_mutation_lock

    monkeypatch.setattr(pipeline, "BACKEND", FixtureBackend())
    run = tmp_path / "run"; config = _config(tmp_path)
    pipeline.init_run(run, config, mode="offline-fixture")
    with stage_mutation_lock(run), pytest.raises(RuntimeError, match="lock is busy"):
        pipeline.execute("emit-corpus-jobs", run, config)


def test_cli_bounds_expected_runtime_errors_without_hiding_programming_bugs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from idea_factory import pipeline
    from idea_factory.cli import main
    from idea_factory.stage_io import stage_mutation_lock

    monkeypatch.setattr(pipeline, "BACKEND", FixtureBackend())
    run = tmp_path / "run"; config = _config(tmp_path)
    pipeline.init_run(run, config, mode="offline-fixture")
    state_before = (run / "state.json").read_bytes()
    files_before = {
        path.relative_to(run).as_posix(): path.read_bytes()
        for path in run.rglob("*") if path.is_file() and path.name != ".stage.lock"
    }
    with stage_mutation_lock(run), pytest.raises(SystemExit) as busy:
        main(["emit-corpus-jobs", "--run", str(run), "--config", str(config)])
    assert busy.value.code == 2
    assert "lock is busy" in capsys.readouterr().err
    assert (run / "state.json").read_bytes() == state_before
    assert {
        path.relative_to(run).as_posix(): path.read_bytes()
        for path in run.rglob("*") if path.is_file() and path.name != ".stage.lock"
    } == files_before

    def expected_compensation(*args, **kwargs):
        raise RuntimeError("finalize persistent ledger compensation is unresolved")

    monkeypatch.setattr(pipeline, "finalize", expected_compensation)
    with pytest.raises(SystemExit) as compensation:
        main(["finalize", "--run", str(run), "--config", str(config)])
    assert compensation.value.code == 2
    assert "compensation is unresolved" in capsys.readouterr().err

    def programming_bug(*args, **kwargs):
        raise RuntimeError("programming bug")

    monkeypatch.setattr(pipeline, "execute", programming_bug)
    with pytest.raises(RuntimeError, match="programming bug"):
        main(["emit-corpus-jobs", "--run", str(run), "--config", str(config)])


def test_status_counts_rejected_external_result_as_unresolved_error(tmp_path: Path) -> None:
    from idea_factory import pipeline

    run = tmp_path / "run"; config = _config(tmp_path)
    pipeline.init_run(run, config, mode="offline-fixture")
    _jsonl(run / "routes" / "outcomes.jsonl", [{"job_id": "route-job", "status": "REJECTED"}])
    assert pipeline.status(run)["counts"]["unresolved-error"] == 1


def test_status_rejects_created_stage_fake_pack_without_side_effects(tmp_path: Path) -> None:
    from idea_factory import pipeline

    run = tmp_path / "run"; config = _config(tmp_path)
    pipeline.init_run(run, config, mode="offline-fixture")
    packs = run / "idea_packs"; packs.mkdir()
    (packs / "manifest.json").write_text(
        json.dumps({"pack_count": 1, "ready_count": 1}), encoding="utf-8",
    )
    before = {
        path.relative_to(run).as_posix(): path.read_bytes()
        for path in run.rglob("*") if path.is_file() and path.name != ".stage.lock"
    }
    current = pipeline.status(run)
    after = {
        path.relative_to(run).as_posix(): path.read_bytes()
        for path in run.rglob("*") if path.is_file() and path.name != ".stage.lock"
    }
    assert after == before
    assert current["idea_pack_artifact_count"] == 0
    assert current["counts"]["surviving"] == 0
    assert current["idea_pack_produced"] is False
    assert current["counts"]["unresolved-error"] > 0


def test_hold_pack_finishes_operationally_without_false_zero_or_idea_yield(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from idea_factory import pipeline
    from idea_factory.models import RunStage, RunState

    class HoldBackend(FixtureBackend):
        def finalize(self, run: Path, **_: object) -> dict[str, object]:
            directory = run / "idea_packs"; directory.mkdir(parents=True, exist_ok=True)
            (directory / "hold.json").write_text(json.dumps({"idea_id": "hold", "status": "HOLD"}), encoding="utf-8")
            (directory / "manifest.json").write_text(json.dumps({"pack_count": 1, "ready_count": 0}), encoding="utf-8")
            return {"pack_count": 1, "ready_count": 0, "ready_packs": []}

    monkeypatch.setattr(pipeline, "BACKEND", HoldBackend())
    run = tmp_path / "run"; config = _config(tmp_path)
    pipeline.init_run(run, config, mode="offline-fixture")
    state = RunState(run_id=json.loads((run / "state.json").read_text(encoding="utf-8"))["run_id"], stage=RunStage.HUMAN_SCORED)
    (run / "state.json").write_text(json.dumps(state.model_dump(mode="json")), encoding="utf-8")
    _write_manual_human_bundle(run)
    assert pipeline.finalize(run, config, allow_test_ready=True) == "COMPLETED"
    result = pipeline.status(run)
    assert result["stage"] == "HUMAN_SCORED"
    assert result["completion_kind"] is None
    assert result["operational_completion"] is True
    assert result["idea_pack_produced"] is False
    assert result["real_idea_pack_produced"] is False


def test_offline_fixture_cannot_self_assert_live_recon(tmp_path: Path) -> None:
    from idea_factory import pipeline

    run = tmp_path / "run"; config = _config(tmp_path)
    pipeline.init_run(run, config, mode="offline-fixture")
    _jsonl(run / "recon" / "ready_for_routes.jsonl", [{"record_id": "opp", "live_ready": True}])
    result = pipeline.status(run)
    assert result["live_recon_executed"] is False
    assert result["real_idea_pack_produced"] is False


def test_init_run_reports_invalid_config_at_cli_boundary(tmp_path: Path) -> None:
    from idea_factory.cli import main

    config = tmp_path / "invalid.json"; config.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["init-run", "--run", str(tmp_path / "run"), "--config", str(config)])
    assert exc.value.code == 2


def test_idempotent_replay_rejects_deleted_owned_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from idea_factory import pipeline

    monkeypatch.setattr(pipeline, "BACKEND", FixtureBackend())
    run = tmp_path / "run"; config = _config(tmp_path)
    pipeline.init_run(run, config, mode="offline-fixture")
    assert pipeline.execute("emit-corpus-jobs", run, config) == "COMPLETED"
    assert pipeline.execute("emit-corpus-jobs", run, config) == "NO_OP"
    (run / "corpus" / "router_jobs.jsonl").unlink()
    with pytest.raises(pipeline.PipelineError, match="owned artifact|missing|new-run"):
        pipeline.execute("emit-corpus-jobs", run, config)
    _jsonl(run / "corpus" / "router_jobs.jsonl", [{"job_id": "tampered"}])
    with pytest.raises(pipeline.PipelineError, match="owned artifact|modified|new-run"):
        pipeline.execute("emit-corpus-jobs", run, config)


def test_status_counts_recon_operator_and_execution_handoffs(tmp_path: Path) -> None:
    from idea_factory import pipeline

    run = tmp_path / "run"; config = _config(tmp_path)
    pipeline.init_run(run, config, mode="offline-fixture")
    _jsonl(run / "recon" / "execution_job_templates.jsonl", [{"template_id": "t1"}, {"template_id": "t2"}])
    assert pipeline.status(run)["counts"]["awaiting-external-result"] == 2
    _jsonl(run / "recon" / "execution_jobs.jsonl", [{"job_id": "j1"}])
    assert pipeline.status(run)["counts"]["awaiting-external-result"] == 1


def test_status_does_not_count_completed_round_two_outcomes_as_awaiting(tmp_path: Path) -> None:
    from idea_factory import pipeline

    run = tmp_path / "run"; config = _config(tmp_path)
    pipeline.init_run(run, config, mode="offline-fixture")
    _jsonl(run / "review" / "round2_jobs.jsonl", [
        {"job_id": "round2-accepted"}, {"job_id": "round2-rejected"},
    ])
    _jsonl(run / "review" / "results.jsonl", [
        {"job_id": "round2-accepted", "round": 2},
    ])
    _jsonl(run / "review" / "outcomes.jsonl", [
        {"job_id": "round2-accepted", "status": "ACCEPTED", "round": 2},
        {"job_id": "round2-rejected", "status": "REJECTED", "round": 2},
    ])
    assert pipeline.status(run)["counts"]["awaiting-external-result"] == 0


def test_status_clears_actual_strict_round_two_after_review_merge(tmp_path: Path) -> None:
    from idea_factory import pipeline
    from idea_factory.artifacts import read_jsonl
    from idea_factory.models import RunStage, RunState
    from idea_factory.review import emit_review_jobs, ingest_review_results, validate_review_bundle
    from test_routes_review import _accepted_routes, _review_result

    run, config = _accepted_routes(tmp_path)
    jobs_path = emit_review_jobs(run, config, allow_test_ready=True)
    jobs = read_jsonl(jobs_path)
    ingest_review_results(
        jobs_path,
        [_review_result(job, "NARROW" if index == 0 else "KILL") for index, job in enumerate(jobs)],
        config,
        allow_test_ready=True,
    )
    pending = emit_review_jobs(run, config, round_number=2, allow_test_ready=True)
    pending_job = read_jsonl(pending)[0]
    ingest_review_results(
        pending, [_review_result(pending_job, "PASS_TO_HUMAN")], config,
        allow_test_ready=True,
    )
    final = validate_review_bundle(run, config, allow_test_ready=True)
    assert any(int(job["round"]) == 2 for job in final["jobs"])
    run_id = "strict-round-two-run"
    (run / "manifest.json").write_text(json.dumps({
        "run_id": run_id, "mode": "offline-fixture",
        "truth_boundary": "OFFLINE_FIXTURE_NOT_LIVE_RECON",
    }), encoding="utf-8")
    state = RunState(run_id=run_id, stage=RunStage.REVIEWED)
    (run / "state.json").write_text(json.dumps(state.model_dump(mode="json")), encoding="utf-8")
    assert pipeline.status(run)["counts"]["awaiting-external-result"] == 0


def test_finalize_state_and_report_publish_roll_back_together(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from idea_factory import pipeline
    from idea_factory.models import RunStage, RunState

    monkeypatch.setattr(pipeline, "BACKEND", FixtureBackend())
    run = tmp_path / "run"; config = _config(tmp_path)
    pipeline.init_run(run, config, mode="offline-fixture")
    state = RunState(run_id=json.loads((run / "state.json").read_text(encoding="utf-8"))["run_id"], stage=RunStage.HUMAN_SCORED)
    (run / "state.json").write_text(json.dumps(state.model_dump(mode="json")), encoding="utf-8")
    _write_manual_human_bundle(run)
    real_publish = pipeline.publish_transaction
    calls = 0

    def fail_once(payloads):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected final publication failure")
        return real_publish(payloads)

    monkeypatch.setattr(pipeline, "publish_transaction", fail_once)
    with pytest.raises(OSError, match="injected final"):
        pipeline.finalize(run, config, allow_test_ready=True)
    persisted = RunState.model_validate(json.loads((run / "state.json").read_text(encoding="utf-8")))
    assert persisted.stage == RunStage.HUMAN_SCORED
    assert "command:finalize" not in persisted.input_hashes
    assert not (run / "report.md").exists()
    assert pipeline.finalize(run, config, allow_test_ready=True) == "COMPLETED"
    assert (run / "report.md").is_file()


def test_real_backend_recon_handoff_materializes_imports_and_reports(tmp_path: Path) -> None:
    import hashlib
    from datetime import datetime, timedelta

    from idea_factory.artifacts import read_jsonl
    from idea_factory.pipeline import RealBackend
    from idea_factory.recon import TERMS_ACKNOWLEDGEMENT, record_terms_notification, validate_recon_bundle
    from test_recon import (
        _arxiv_paper,
        _build_dedup_run,
        _fake_skill_roots,
        _fake_uv,
        _operator_attestation_body,
        _report_result,
    )

    run, config = _build_dedup_run(tmp_path)
    backend = RealBackend()
    assert backend.emit_recon_pack(run, config=config, recon_context=None) is False

    record_terms_notification(
        tmp_path,
        user_notified_at="2026-08-03T09:00:00+08:00",
        acknowledgement=TERMS_ACKNOWLEDGEMENT,
    )
    roots = _fake_skill_roots(tmp_path)
    uv = _fake_uv(tmp_path)
    context_file = tmp_path / "recon-context.json"
    context_file.write_text(json.dumps({
        "workspace_root": str(tmp_path.resolve()),
        "skill_roots": {key: str(value.resolve()) for key, value in roots.items()},
        "prepared_at": "2026-08-03T09:05:00+08:00",
        "uv_executable": str(uv.resolve()),
        "allow_test_attestation": True,
        "operator_attestation": _operator_attestation_body(tmp_path, roots, uv),
    }), encoding="utf-8")
    assert backend.emit_recon_pack(run, config=config, recon_context=context_file) is True
    jobs = read_jsonl(run / "recon" / "execution_jobs.jsonl")
    assert jobs

    execution_context = json.loads((run / "recon" / "execution_context.json").read_text(encoding="utf-8"))
    external = tmp_path / "execution-bundle"; external.mkdir()
    receipts = []
    arxiv_index = 0
    omitted_error_job = next(job["job_id"] for job in jobs if job["source"] == "OPENALEX")
    base_time = datetime.fromisoformat("2026-08-03T10:00:00+08:00")
    for job in jobs:
        if job["source"] == "ARXIV":
            paper = _arxiv_paper("http://arxiv.org/abs/2501.00001v1", "Bound nearest prior", "https://arxiv.org/pdf/2501.00001")
            raw = json.dumps({"status": "success", "results_count": 1, "papers": [paper]}).encode()
        else:
            raw = b'{"meta":{"count":0,"per_page":10},"results":[]}'
        relative = Path(str(job["raw_output_path"]))
        target = external.joinpath(*relative.parts[1:])
        if job["job_id"] != omitted_error_job:
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)
        request_time = base_time
        if job["source"] == "ARXIV":
            request_time += timedelta(seconds=3 * arxiv_index); arxiv_index += 1
        receipts.append({
            "schema_version": "idea_factory.recon_execution_receipt.v3",
            "job_id": job["job_id"], "source": job["source"], "query_id": job["query_id"],
            "argv_sha256": job["argv_sha256"], "raw_output_path": job["raw_output_path"],
            "query_pack_sha256": job["query_pack_sha256"],
            "command_policy_version": job["command_policy_version"],
            "started_at": request_time.isoformat(), "request_started_at": request_time.isoformat(),
            "completed_at": (request_time + timedelta(seconds=1)).isoformat(),
            "exit_code": 1 if job["job_id"] == omitted_error_job else 0,
            "status": "ERROR" if job["job_id"] == omitted_error_job else ("SUCCESS" if job["source"] == "ARXIV" else "EMPTY"),
            "http_status": None if job["job_id"] == omitted_error_job else 200,
            "raw_file_sha256": None if job["job_id"] == omitted_error_job else hashlib.sha256(raw).hexdigest(),
            "execution_context_sha256": execution_context["execution_context_sha256"],
            "tool_stdout_summary": "not executed" if job["job_id"] == omitted_error_job else ("one result" if job["source"] == "ARXIV" else "empty result"),
            "tool_stderr_summary": "not executed" if job["job_id"] == omitted_error_job else "",
            "error_reason": "optional source unavailable" if job["job_id"] == omitted_error_job else "",
        })
    _jsonl(external / "execution_receipts.jsonl", receipts)

    assert backend.ingest_recon(run, config=config, results=None, execution_bundle=external) is False
    report_jobs = read_jsonl(run / "recon" / "report_jobs.jsonl")
    assert report_jobs
    report_results = tmp_path / "report-results.jsonl"
    report_results.write_text(
        "".join(_report_result(job, "NO_DIRECT_COVERAGE_FOUND", evidence=job["evidence"][0], include_prior=False) + "\n" for job in report_jobs),
        encoding="utf-8",
    )
    assert backend.ingest_recon(run, config=config, results=report_results, execution_bundle=None) is True
    bundle = validate_recon_bundle(run, config)
    assert len(bundle["reports"]) == len(report_jobs)


def test_malformed_recon_raw_is_compensated_and_remains_visible_for_retry(tmp_path: Path) -> None:
    import hashlib

    from idea_factory.artifacts import read_jsonl
    from idea_factory.models import RunStage, RunState
    from idea_factory.pipeline import RealBackend
    from test_recon import _build_dedup_run, _emit_empty_execution_chain

    run, config = _build_dedup_run(tmp_path)
    jobs, receipts = _emit_empty_execution_chain(run, config)
    external = tmp_path / "malformed-execution"; external.mkdir()
    for job in jobs:
        source = run / str(job["raw_output_path"])
        relative = Path(str(job["raw_output_path"]))
        target = external.joinpath(*relative.parts[1:])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        source.unlink()
    (run / "recon" / "execution_receipts.jsonl").unlink()
    malformed_job = next(job for job in jobs if job["source"] == "OPENALEX")
    malformed_target = external.joinpath(*Path(str(malformed_job["raw_output_path"])).parts[1:])
    malformed = b'{"meta":'
    malformed_target.write_bytes(malformed)
    for receipt in receipts:
        if receipt["job_id"] == malformed_job["job_id"]:
            receipt["raw_file_sha256"] = hashlib.sha256(malformed).hexdigest()
    _jsonl(external / "execution_receipts.jsonl", receipts)
    run_id = "malformed-recon-run"
    (run / "manifest.json").write_text(json.dumps({
        "run_id": run_id, "mode": "offline-fixture",
        "truth_boundary": "OFFLINE_FIXTURE_NOT_LIVE_RECON",
    }), encoding="utf-8")
    state = RunState(run_id=run_id, stage=RunStage.QUALITY_GATED)
    (run / "state.json").write_text(json.dumps(state.model_dump(mode="json")), encoding="utf-8")

    backend = RealBackend()
    with pytest.raises(ValueError, match="JSON|OpenAlex|raw|parse"):
        backend.ingest_recon(run, config=config, results=None, execution_bundle=external)
    assert not (run / "recon" / "execution_receipts.jsonl").exists()
    assert all(not (run / str(job["raw_output_path"])).exists() for job in jobs)
    current = __import__("idea_factory.pipeline", fromlist=["status"]).status(run)
    assert current["counts"]["completed"] == 0
    assert current["counts"]["awaiting-external-result"] == len(jobs)
    assert current["counts"]["unresolved-error"] == 0

    repaired = b'{"meta":{"count":0,"per_page":10},"results":[]}'
    malformed_target.write_bytes(repaired)
    for receipt in receipts:
        if receipt["job_id"] == malformed_job["job_id"]:
            receipt["raw_file_sha256"] = hashlib.sha256(repaired).hexdigest()
    _jsonl(external / "execution_receipts.jsonl", receipts)
    assert backend.ingest_recon(
        run, config=config, results=None, execution_bundle=external,
    ) is False
    assert read_jsonl(run / "recon" / "report_jobs.jsonl")


def test_route_and_review_prompt_or_upstream_drift_requires_new_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from idea_factory import pipeline
    from idea_factory.models import RunStage, RunState

    monkeypatch.setattr(pipeline, "BACKEND", FixtureBackend())
    config = _config(tmp_path)
    route_prompt = tmp_path / "route.md"; route_prompt.write_text("route v1", encoding="utf-8")
    review_prompt = tmp_path / "review.md"; review_prompt.write_text("review v1", encoding="utf-8")

    route_run = tmp_path / "route-run"
    pipeline.init_run(route_run, config, mode="offline-fixture")
    route_state = RunState(
        run_id=json.loads((route_run / "state.json").read_text(encoding="utf-8"))["run_id"],
        stage=RunStage.RECON_READY,
    )
    (route_run / "state.json").write_text(json.dumps(route_state.model_dump(mode="json")), encoding="utf-8")
    _write_manual_recon_bundle(route_run)
    assert pipeline.execute(
        "emit-route-jobs", route_run, config, route_prompt=route_prompt, allow_test_ready=True,
    ) == "COMPLETED"
    route_prompt.write_text("route v2", encoding="utf-8")
    with pytest.raises(pipeline.PipelineError, match="input hashes changed|new-run"):
        pipeline.execute("emit-route-jobs", route_run, config, route_prompt=route_prompt, allow_test_ready=True)
    route_prompt.write_text("route v1", encoding="utf-8")
    _jsonl(route_run / "recon" / "reports.jsonl", [{"drift": True}])
    with pytest.raises(pipeline.PipelineError, match="input hashes changed|new-run"):
        pipeline.execute("emit-route-jobs", route_run, config, route_prompt=route_prompt, allow_test_ready=True)

    review_run = tmp_path / "review-run"
    pipeline.init_run(review_run, config, mode="offline-fixture")
    review_state = RunState(
        run_id=json.loads((review_run / "state.json").read_text(encoding="utf-8"))["run_id"],
        stage=RunStage.ROUTES_READY,
    )
    (review_run / "state.json").write_text(json.dumps(review_state.model_dump(mode="json")), encoding="utf-8")
    _write_manual_route_bundle(review_run)
    assert pipeline.execute(
        "emit-review-jobs", review_run, config, route_prompt=route_prompt,
        review_prompt=review_prompt, allow_test_ready=True,
    ) == "COMPLETED"
    review_prompt.write_text("review v2", encoding="utf-8")
    with pytest.raises(pipeline.PipelineError, match="input hashes changed|new-run"):
        pipeline.execute(
            "emit-review-jobs", review_run, config, route_prompt=route_prompt,
            review_prompt=review_prompt, allow_test_ready=True,
        )


def test_finalize_repository_root_drift_requires_new_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from idea_factory import pipeline
    from idea_factory.models import RunStage, RunState

    monkeypatch.setattr(pipeline, "BACKEND", FixtureBackend())
    run = tmp_path / "run"; config = _config(tmp_path)
    pipeline.init_run(run, config, mode="offline-fixture")
    state = RunState(
        run_id=json.loads((run / "state.json").read_text(encoding="utf-8"))["run_id"],
        stage=RunStage.HUMAN_SCORED,
    )
    (run / "state.json").write_text(json.dumps(state.model_dump(mode="json")), encoding="utf-8")
    _write_manual_human_bundle(run)
    first_root = tmp_path / "repo-a"; first_root.mkdir()
    second_root = tmp_path / "repo-b"; second_root.mkdir()
    assert pipeline.finalize(
        run, config, allow_test_ready=True, repository_root=first_root,
    ) == "COMPLETED"
    with pytest.raises(pipeline.PipelineError, match="input hashes changed|new-run"):
        pipeline.finalize(run, config, allow_test_ready=True, repository_root=second_root)


@pytest.mark.parametrize("preexisting", [False, True])
def test_real_finalize_publication_failure_removes_only_failed_run_ledger_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preexisting: bool,
) -> None:
    from idea_factory import pipeline
    from idea_factory.artifacts import read_jsonl
    from idea_factory.ledger import validate_ledger_updates

    repository = tmp_path / "repository"; repository.mkdir()
    persistent_before = _seed_persistent_ledger(tmp_path, repository) if preexisting else None
    target_a = tmp_path / "target-a"; target_a.mkdir()
    target_b = tmp_path / "target-b"; target_b.mkdir()
    run_a, config_a, resolved_a = _real_human_scored_pipeline_run(target_a)
    run_b, config_b, resolved_b = _real_human_scored_pipeline_run(target_b)
    state_a_before = (run_a / "state.json").read_bytes()
    assert not (run_a / "idea_packs").exists()
    assert not (run_a / "ledger" / "updates.jsonl").exists()
    real_publish = pipeline.publish_transaction
    interleaved = False
    persistent_after_b: list[dict[str, object]] | None = None

    def finish_b_then_fail_a(payloads):
        nonlocal interleaved, persistent_after_b
        if run_a / "state.json" in payloads and not interleaved:
            interleaved = True
            assert pipeline.finalize(
                run_b, config_b, allow_test_ready=True, repository_root=repository,
            ) == "COMPLETED"
            persistent_after_b = read_jsonl(repository / "data" / "idea_ledger.jsonl")
            raise OSError("injected outer finalize publication failure")
        return real_publish(payloads)

    monkeypatch.setattr(pipeline, "publish_transaction", finish_b_then_fail_a)
    with pytest.raises(OSError, match="outer finalize"):
        pipeline.finalize(
            run_a, config_a, allow_test_ready=True, repository_root=repository,
        )
    assert interleaved and persistent_after_b is not None
    assert (run_a / "state.json").read_bytes() == state_a_before
    assert not (run_a / "report.md").exists()
    assert not (run_a / "idea_packs").exists()
    assert not (run_a / "ledger" / "updates.jsonl").exists()
    persistent = repository / "data" / "idea_ledger.jsonl"
    failed_run_id = json.loads(state_a_before)["run_id"]
    expected_after_rollback = [
        row for row in persistent_after_b if row["run_id"] != failed_run_id
    ]
    assert read_jsonl(persistent) == expected_after_rollback
    if persistent_before is not None:
        prior_rows = [json.loads(line) for line in persistent_before.decode("utf-8").splitlines()]
        assert all(row in expected_after_rollback for row in prior_rows)
    validate_ledger_updates(
        run_b, resolved_b, repository_root=repository, allow_test_ready=True,
    )
    assert pipeline.finalize(
        run_a, config_a, allow_test_ready=True, repository_root=repository,
    ) == "COMPLETED"
    validate_ledger_updates(
        run_a, resolved_a, repository_root=repository, allow_test_ready=True,
    )
    validate_ledger_updates(
        run_b, resolved_b, repository_root=repository, allow_test_ready=True,
    )
    persistent_final = read_jsonl(persistent)
    b_run_id = json.loads((run_b / "state.json").read_text(encoding="utf-8"))["run_id"]
    assert [row for row in persistent_final if row["run_id"] == b_run_id] == [
        row for row in expected_after_rollback if row["run_id"] == b_run_id
    ]


def test_persistent_compensation_conflict_preserves_shared_ledger_and_records_unresolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from idea_factory import pipeline
    from idea_factory.artifacts import read_jsonl

    repository = tmp_path / "repository"; repository.mkdir()
    target = tmp_path / "target"; target.mkdir()
    run, config, _resolved = _real_human_scored_pipeline_run(target)
    real_publish = pipeline.publish_transaction
    conflicted_bytes: bytes | None = None

    def conflict_then_fail(payloads):
        nonlocal conflicted_bytes
        if run / "state.json" in payloads and conflicted_bytes is None:
            persistent = repository / "data" / "idea_ledger.jsonl"
            rows = read_jsonl(persistent)
            rows[0]["status"] = "CONCURRENT_CONFLICT"
            _jsonl(persistent, rows)
            conflicted_bytes = persistent.read_bytes()
            raise OSError("injected outer finalize publication failure")
        return real_publish(payloads)

    monkeypatch.setattr(pipeline, "publish_transaction", conflict_then_fail)
    with pytest.raises(RuntimeError, match="persistent ledger compensation"):
        pipeline.finalize(
            run, config, allow_test_ready=True, repository_root=repository,
        )
    persistent = repository / "data" / "idea_ledger.jsonl"
    assert conflicted_bytes is not None
    assert persistent.read_bytes() == conflicted_bytes
    state = json.loads((run / "state.json").read_text(encoding="utf-8"))
    assert state["unresolved_errors"] == [
        "FINALIZE_PERSISTENT_LEDGER_COMPENSATION_UNRESOLVED"
    ]
    assert pipeline.status(run)["counts"]["unresolved-error"] == 1


@pytest.mark.parametrize("marker_fails", [False, True])
def test_local_compensation_failure_preserves_original_and_records_durable_unresolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, marker_fails: bool,
) -> None:
    from idea_factory import pipeline
    from idea_factory.models import RunStage, RunState

    class OriginalFailureBackend(FixtureBackend):
        def finalize(self, run: Path, **_: object) -> dict[str, object]:
            packs = run / "idea_packs"; packs.mkdir(parents=True, exist_ok=True)
            (packs / "residual.json").write_text(
                json.dumps({"idea_id": "residual", "status": "READY_FOR_CHEAP_TEST"}),
                encoding="utf-8",
            )
            (packs / "manifest.json").write_text(
                json.dumps({"pack_count": 1, "ready_count": 1}), encoding="utf-8",
            )
            raise RuntimeError("ORIGINAL finalize failure")

    monkeypatch.setattr(pipeline, "BACKEND", OriginalFailureBackend())
    run = tmp_path / "run"; config = _config(tmp_path)
    pipeline.init_run(run, config, mode="offline-fixture")
    initial = RunState(
        run_id=json.loads((run / "state.json").read_text(encoding="utf-8"))["run_id"],
        stage=RunStage.HUMAN_SCORED,
    )
    (run / "state.json").write_text(
        json.dumps(initial.model_dump(mode="json")), encoding="utf-8",
    )
    _write_manual_human_bundle(run)

    def local_restore_failure(*args, **kwargs):
        raise RuntimeError("LOCAL restore failure")

    monkeypatch.setattr(
        pipeline, "_restore_finalization_local_snapshot", local_restore_failure,
    )
    if marker_fails:
        real_publish = pipeline.publish_transaction

        def durable_marker_failure(payloads):
            state_payload = payloads.get(run / "state.json")
            if state_payload is not None and b"FINALIZE_LOCAL_COMPENSATION_UNRESOLVED" in state_payload:
                raise OSError("MARKER write failure")
            return real_publish(payloads)

        monkeypatch.setattr(pipeline, "publish_transaction", durable_marker_failure)

    with pytest.raises(RuntimeError) as captured:
        pipeline.finalize(run, config, allow_test_ready=True)
    message = str(captured.value)
    assert "ORIGINAL finalize failure" in message
    assert "LOCAL restore failure" in message
    if marker_fails:
        assert "durable" in message and "MARKER write failure" in message
        return

    state = json.loads((run / "state.json").read_text(encoding="utf-8"))
    assert state["unresolved_errors"] == ["FINALIZE_LOCAL_COMPENSATION_UNRESOLVED"]
    current = pipeline.status(run)
    assert current["counts"]["unresolved-error"] > 0
    assert current["counts"]["surviving"] == 0
    assert current["idea_pack_produced"] is False
    assert current["completion_kind"] is None


def test_complete_finalize_replay_officially_revalidates_and_rejects_persistent_ledger_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from idea_factory import ledger as ledger_module
    from idea_factory import pipeline, render as render_module
    from idea_factory.artifacts import read_jsonl

    repository = tmp_path / "repository"; repository.mkdir()
    target = tmp_path / "target"; target.mkdir()
    run, config, _resolved = _real_human_scored_pipeline_run(target)
    assert pipeline.finalize(
        run, config, allow_test_ready=True, repository_root=repository,
    ) == "COMPLETED"
    real_ledger_validator = ledger_module.validate_ledger_updates
    real_pack_validator = render_module.validate_idea_pack_bundle
    calls = {"ledger": 0, "packs": 0}

    def ledger_spy(*args, **kwargs):
        calls["ledger"] += 1
        return real_ledger_validator(*args, **kwargs)

    def pack_spy(*args, **kwargs):
        calls["packs"] += 1
        return real_pack_validator(*args, **kwargs)

    monkeypatch.setattr(ledger_module, "validate_ledger_updates", ledger_spy)
    monkeypatch.setattr(render_module, "validate_idea_pack_bundle", pack_spy)
    assert pipeline.finalize(
        run, config, allow_test_ready=True, repository_root=repository,
    ) == "NO_OP"
    assert calls["ledger"] >= 1 and calls["packs"] >= 1

    persistent = repository / "data" / "idea_ledger.jsonl"
    original = persistent.read_bytes()
    rows = read_jsonl(persistent)
    rows[0]["status"] = "TAMPERED"
    _jsonl(persistent, rows)
    with pytest.raises((pipeline.PipelineError, ValueError), match="persistent|ledger|modified"):
        pipeline.finalize(run, config, allow_test_ready=True, repository_root=repository)
    persistent.write_bytes(original)
    assert pipeline.finalize(
        run, config, allow_test_ready=True, repository_root=repository,
    ) == "NO_OP"
    persistent.unlink()
    with pytest.raises((pipeline.PipelineError, ValueError), match="persistent|ledger|missing"):
        pipeline.finalize(run, config, allow_test_ready=True, repository_root=repository)


def test_status_rejects_tampered_complete_pack_without_side_effects(tmp_path: Path) -> None:
    from idea_factory import pipeline

    repository = tmp_path / "repository"; repository.mkdir()
    target = tmp_path / "target"; target.mkdir()
    run, config, _resolved = _real_human_scored_pipeline_run(target)
    assert pipeline.finalize(
        run, config, allow_test_ready=True, repository_root=repository,
    ) == "COMPLETED"
    manifest = json.loads(
        (run / "idea_packs" / "manifest.json").read_text(encoding="utf-8"),
    )
    pack = run / "idea_packs" / f"{manifest['idea_ids'][0]}.json"
    payload = json.loads(pack.read_text(encoding="utf-8"))
    payload["source_of_gain"] = "tampered after completion"
    pack.write_text(json.dumps(payload), encoding="utf-8")
    before = {
        path.relative_to(run).as_posix(): path.read_bytes()
        for path in run.rglob("*") if path.is_file() and path.name != ".stage.lock"
    }
    current = pipeline.status(run)
    after = {
        path.relative_to(run).as_posix(): path.read_bytes()
        for path in run.rglob("*") if path.is_file() and path.name != ".stage.lock"
    }
    assert after == before
    assert current["idea_pack_artifact_count"] == 0
    assert current["counts"]["surviving"] == 0
    assert current["idea_pack_produced"] is False
    assert current["counts"]["unresolved-error"] > 0


def test_status_replays_exact_custom_finalize_prompts_and_rejects_prompt_drift(
    tmp_path: Path,
) -> None:
    from idea_factory import pipeline
    from idea_factory.models import CompletionManifest, IdeaPack, RunState

    repository = tmp_path / "repository"; repository.mkdir()
    target = tmp_path / "target"; target.mkdir()
    custom_route = tmp_path / "custom-route.md"
    custom_review = tmp_path / "custom-review.md"
    route_bytes = pipeline.DEFAULT_PROMPTS["route"].read_bytes()
    review_bytes = pipeline.DEFAULT_PROMPTS["review"].read_bytes()
    custom_route.write_bytes(route_bytes)
    custom_review.write_bytes(review_bytes)
    run, config, _resolved = _real_human_scored_pipeline_run(
        target, route_prompt=custom_route, review_prompt=custom_review,
    )
    assert pipeline.finalize(
        run, config, allow_test_ready=True, repository_root=repository,
        route_prompt=custom_route, review_prompt=custom_review,
    ) == "COMPLETED"
    current = pipeline.status(run)
    assert current["stage"] == "COMPLETE"
    assert current["counts"]["surviving"] == 1
    assert current["counts"]["unresolved-error"] == 0
    assert current["idea_pack_produced"] is True
    assert current["live_recon_executed"] is False
    assert current["real_idea_pack_produced"] is False
    assert current["cheap_test_executed"] is False

    custom_route.write_bytes(route_bytes + b"\nchanged\n")
    drifted = pipeline.status(run)
    assert drifted["counts"]["surviving"] == 0
    assert drifted["counts"]["unresolved-error"] > 0
    assert drifted["idea_pack_produced"] is False
    with pytest.raises(pipeline.PipelineError, match="new-run"):
        pipeline.finalize(
            run, config, allow_test_ready=True, repository_root=repository,
            route_prompt=custom_route, review_prompt=custom_review,
        )
    custom_route.write_bytes(route_bytes)
    assert pipeline.status(run)["idea_pack_produced"] is True

    custom_review.unlink()
    missing = pipeline.status(run)
    assert missing["counts"]["surviving"] == 0
    assert missing["counts"]["unresolved-error"] > 0
    assert missing["idea_pack_produced"] is False
    custom_review.write_bytes(review_bytes)
    assert pipeline.status(run)["idea_pack_produced"] is True

    state = RunState.model_validate_json((run / "state.json").read_text(encoding="utf-8"))
    original_state_bytes = (run / "state.json").read_bytes()
    records = state.completion_evidence.artifact.records()
    packs = tuple(IdeaPack.model_validate(record["idea_pack"]) for record in records)
    stripped_hashes = {
        key: value for key, value in state.input_hashes.items()
        if not key.startswith("external-input:finalize:")
    }
    stripped_evidence = CompletionManifest.from_ready_packs(
        run_id=state.run_id, input_hashes=stripped_hashes, packs=packs,
    )
    stripped_state = state.model_dump(mode="json")
    stripped_state["input_hashes"] = stripped_hashes
    stripped_state["completion_evidence"] = stripped_evidence.model_dump(mode="json")
    (run / "state.json").write_text(json.dumps(stripped_state), encoding="utf-8")
    no_fallback = pipeline.status(run)
    assert no_fallback["counts"]["surviving"] == 0
    assert no_fallback["counts"]["unresolved-error"] > 0
    assert no_fallback["idea_pack_produced"] is False
    (run / "state.json").write_bytes(original_state_bytes)

    state = RunState.model_validate_json((run / "state.json").read_text(encoding="utf-8"))
    tampered_route = tmp_path / "tampered-route.md"
    tampered_route.write_bytes(route_bytes)
    tampered_hashes = dict(state.input_hashes)
    tampered_hashes["external-input:finalize:route-prompt-path"] = str(tampered_route.resolve())
    tampered_evidence = CompletionManifest.from_ready_packs(
        run_id=state.run_id, input_hashes=tampered_hashes, packs=packs,
    )
    raw_state = state.model_dump(mode="json")
    raw_state["input_hashes"] = tampered_hashes
    raw_state["completion_evidence"] = tampered_evidence.model_dump(mode="json")
    (run / "state.json").write_text(json.dumps(raw_state), encoding="utf-8")
    path_tampered = pipeline.status(run)
    assert path_tampered["counts"]["surviving"] == 0
    assert path_tampered["counts"]["unresolved-error"] > 0
    assert path_tampered["idea_pack_produced"] is False


def test_status_safely_replays_legacy_default_finalize_without_prompt_bindings(
    tmp_path: Path,
) -> None:
    from idea_factory import pipeline
    from idea_factory.models import CompletionManifest, IdeaPack, RunState

    repository = tmp_path / "repository"; repository.mkdir()
    target = tmp_path / "target"; target.mkdir()
    run, config, _resolved = _real_human_scored_pipeline_run(target)
    assert pipeline.finalize(
        run, config, allow_test_ready=True, repository_root=repository,
    ) == "COMPLETED"
    state = RunState.model_validate_json((run / "state.json").read_text(encoding="utf-8"))
    legacy_hashes = {
        key: value for key, value in state.input_hashes.items()
        if not key.startswith("external-input:finalize:")
    }
    records = state.completion_evidence.artifact.records()
    packs = tuple(IdeaPack.model_validate(record["idea_pack"]) for record in records)
    legacy_evidence = CompletionManifest.from_ready_packs(
        run_id=state.run_id, input_hashes=legacy_hashes, packs=packs,
    )
    raw_state = state.model_dump(mode="json")
    raw_state["input_hashes"] = legacy_hashes
    raw_state["completion_evidence"] = legacy_evidence.model_dump(mode="json")
    (run / "state.json").write_text(json.dumps(raw_state), encoding="utf-8")
    current = pipeline.status(run)
    assert current["counts"]["surviving"] == 1
    assert current["counts"]["unresolved-error"] == 0
    assert current["idea_pack_produced"] is True


@pytest.mark.parametrize("allow_test_ready", [False, True])
def test_zero_survivor_status_uses_exact_finalize_allow_test_ready_binding(
    tmp_path: Path, allow_test_ready: bool,
) -> None:
    from idea_factory import pipeline

    repository = tmp_path / "repository"; repository.mkdir()
    target = tmp_path / "target"; target.mkdir()
    run, config, _resolved = _real_zero_survivor_pipeline_run(
        target, allow_test_ready=allow_test_ready,
    )
    assert pipeline.finalize(
        run, config, allow_test_ready=allow_test_ready, repository_root=repository,
    ) == "COMPLETED"
    current = pipeline.status(run)
    assert current["stage"] == "COMPLETE"
    assert current["completion_kind"] == "AUDITABLE_ZERO_SURVIVOR"
    assert current["counts"]["completed"] == 1
    assert current["counts"]["surviving"] == 0
    assert current["counts"]["unresolved-error"] == 0
    assert current["idea_pack_produced"] is False


@pytest.mark.parametrize("allow_test_ready", [False, True])
def test_finalize_option_binding_tamper_and_legacy_recovery_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, allow_test_ready: bool,
) -> None:
    from idea_factory import pipeline
    from idea_factory.models import CompletionManifest, RunStage, RunState

    repository = tmp_path / "repository"; repository.mkdir()
    target = tmp_path / "target"; target.mkdir()
    run, config, _resolved = _real_zero_survivor_pipeline_run(
        target, allow_test_ready=allow_test_ready,
    )
    assert pipeline.finalize(
        run, config, allow_test_ready=allow_test_ready, repository_root=repository,
    ) == "COMPLETED"
    state = RunState.model_validate_json((run / "state.json").read_text(encoding="utf-8"))
    version_key = "external-input:finalize:binding-version"
    option_key = "external-input:finalize:option:allow-test-ready"
    assert state.input_hashes[version_key] == "idea_factory.finalize_binding.v1"
    assert state.input_hashes[option_key] == ("true" if allow_test_ready else "false")

    def write_hashes(hashes: dict[str, str]) -> None:
        evidence = CompletionManifest.from_gate(
            run_id=state.run_id, gate_stage=RunStage.REVIEWED, input_hashes=hashes,
        )
        raw = state.model_dump(mode="json")
        raw["input_hashes"] = hashes
        raw["completion_evidence"] = evidence.model_dump(mode="json")
        (run / "state.json").write_text(json.dumps(raw), encoding="utf-8")

    for mutation in ("not-a-bool", None):
        hashes = dict(state.input_hashes)
        if mutation is None:
            hashes.pop(option_key)
        else:
            hashes[option_key] = mutation
        write_hashes(hashes)
        invalid = pipeline.status(run)
        assert invalid["counts"]["unresolved-error"] > 0
        assert invalid["counts"]["surviving"] == 0

    missing_family = dict(state.input_hashes)
    missing_family.pop(version_key)
    write_hashes(missing_family)
    assert pipeline.status(run)["counts"]["unresolved-error"] > 0

    legacy = dict(state.input_hashes)
    legacy.pop(version_key)
    legacy.pop(option_key)
    write_hashes(legacy)
    recovered = pipeline.status(run)
    assert recovered["counts"]["unresolved-error"] == 0
    assert recovered["completion_kind"] == "AUDITABLE_ZERO_SURVIVOR"

    no_match = dict(legacy)
    no_match["command:finalize"] = "0" * 64
    write_hashes(no_match)
    assert pipeline.status(run)["counts"]["unresolved-error"] > 0

    write_hashes(legacy)
    stored_digest = legacy["command:finalize"]
    monkeypatch.setattr(pipeline, "_command_digest", lambda *args, **kwargs: stored_digest)
    ambiguous = pipeline.status(run)
    assert ambiguous["counts"]["unresolved-error"] > 0
    assert ambiguous["counts"]["surviving"] == 0


def test_real_cli_offline_fixture_runs_complete_strict_pipeline_without_backend_replacement(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    import hashlib
    from datetime import datetime, timedelta

    from idea_factory.artifacts import read_jsonl
    from idea_factory.recon import TERMS_ACKNOWLEDGEMENT, record_terms_notification
    from test_cards import _card, _result
    from test_ledger import _human_score
    from test_opportunities import _empty_result
    from test_recon import (
        _arxiv_paper, _fake_skill_roots, _fake_uv,
        _operator_attestation_body, _report_result,
    )
    from test_routes_review import _review_result, _route_result

    notes = tmp_path / "notes"; notes.mkdir()
    papers = tmp_path / "papers"; papers.mkdir()
    note_paths = []
    for name in ("kv", "memory"):
        note = notes / f"{name}.md"
        note.write_text(f"# {name}\nObserved cache retrieval failure in long serving.", encoding="utf-8")
        note_paths.append(note.resolve())
    candidates = papers / "list.txt"
    candidates.write_text("".join(str(path) + "\n" for path in note_paths), encoding="utf-8")
    legacy = papers / "idea_ledger.md"; legacy.write_text("legacy\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "notes_root": "notes", "candidate_lists": ["papers/list.txt"],
        "legacy_ledger": "papers/idea_ledger.md", "target_min": 2, "target_max": 2,
        "allowed_labels": ["KV_CACHE", "LONG_MEMORY"], "bridge_regression_slugs": [],
    }), encoding="utf-8")
    run = tmp_path / "run"
    assert _invoke("init-run", "--run", str(run), "--config", str(config), "--mode", "offline-fixture") == 0

    assert _invoke("emit-corpus-jobs", "--run", str(run), "--config", str(config)) == 0
    router_jobs = read_jsonl(run / "corpus" / "router_jobs.jsonl")
    labels = tmp_path / "labels.jsonl"
    _jsonl(labels, [{
        "schema_version": "idea_factory.corpus_router_result.v2",
        **{key: job[key] for key in ("job_id", "slug", "note_sha256", "prompt_sha256")},
        "label": "KV_CACHE" if index == 0 else "LONG_MEMORY",
        "core_mechanism": "bounded memory eviction", "scope_reason": "KV-memory serving",
        "evidence_locator": "note:2", "confidence": "HIGH",
    } for index, job in enumerate(router_jobs)])
    assert _invoke("ingest-corpus-labels", "--run", str(run), "--config", str(config), "--results", str(labels)) == 0

    assert _invoke("emit-card-jobs", "--run", str(run), "--config", str(config)) == 0
    card_jobs = read_jsonl(run / "cards" / "card_jobs.jsonl")
    cards = tmp_path / "cards.jsonl"
    _jsonl(cards, [
        _result(job, [_card(job, f"{job['slug']}-0"), _card(job, f"{job['slug']}-1")])
        for job in card_jobs
    ])
    assert _invoke("ingest-cards", "--run", str(run), "--config", str(config), "--results", str(cards)) == 0
    assert _invoke("build-landscape", "--run", str(run), "--config", str(config)) == 0
    assert _invoke("emit-opportunity-jobs", "--run", str(run), "--config", str(config)) == 0
    opportunity_jobs = read_jsonl(run / "opportunities" / "mining_jobs.jsonl")
    assert opportunity_jobs
    opportunity_results = []
    for index, job in enumerate(opportunity_jobs):
        wrapper = json.loads(_empty_result(job))
        if index == 0:
            wrapper["opportunities"] = [{
                "schema_version": "idea_factory.opportunity.v1",
                "opportunity_id": "opp-real-cli", "operator": job["operator"],
                "assumption_x": "The cache remains resident during long serving traces.",
                "observation_y": "Cards report retrieval degrades after eviction.",
                "condition_z": "under long serving traces",
                "failure_f": "cache eviction causes retrieval loss",
                "missing_capability_w": "eviction-aware retrieval control",
                "alternative_explanation_a": "cache eviction",
                "decisive_experiment": "Compare recall after forced cache eviction against cache eviction.",
                "supporting_card_ids": [job["card_ids"][0]],
                "nearest_internal_neighbors": [job["neighbor_ids"][0]],
                "scope_compatibility": "Both cards cover KV-memory serving over long traces.",
                "inference_flags": ["failure mechanism inferred"],
            }]
        opportunity_results.append(wrapper)
    opportunities = tmp_path / "opportunities.jsonl"; _jsonl(opportunities, opportunity_results)
    assert _invoke("ingest-opportunities", "--run", str(run), "--config", str(config), "--results", str(opportunities)) == 0
    assert _invoke("quality-gate", "--run", str(run), "--config", str(config)) == 0

    assert _invoke("internal-dedup", "--run", str(run), "--config", str(config)) == 0
    shape_jobs = read_jsonl(run / "ledger" / "shape_assignment_jobs.jsonl")
    shapes = tmp_path / "shapes.jsonl"
    _jsonl(shapes, [{
        "schema_version": "idea_factory.shape_assignment_result.v1",
        **{key: job[key] for key in (
            "job_id", "opportunity_id", "quality_record_sha256",
            "quality_bundle_sha256", "legacy_index_sha256",
        )},
        "old_assumption": "cache remains resident",
        "failure_mechanism": "cache eviction causes retrieval loss",
        "missing_capability": "eviction-aware retrieval control",
        "target_scope": "KV-memory serving", "mechanism_family": "EVICTION_CONTROL",
        "residual_difference": "",
    } for job in shape_jobs])
    assert _invoke("internal-dedup", "--run", str(run), "--config", str(config), "--results", str(shapes)) == 0

    assert _invoke("emit-recon-pack", "--run", str(run), "--config", str(config)) == 0
    record_terms_notification(
        tmp_path, user_notified_at="2026-08-03T09:00:00+08:00",
        acknowledgement=TERMS_ACKNOWLEDGEMENT,
    )
    roots = _fake_skill_roots(tmp_path); uv = _fake_uv(tmp_path)
    context = tmp_path / "recon-context.json"
    context.write_text(json.dumps({
        "workspace_root": str(tmp_path.resolve()),
        "skill_roots": {key: str(value.resolve()) for key, value in roots.items()},
        "prepared_at": "2026-08-03T09:05:00+08:00", "uv_executable": str(uv.resolve()),
        "allow_test_attestation": True,
        "operator_attestation": _operator_attestation_body(tmp_path, roots, uv),
    }), encoding="utf-8")
    assert _invoke(
        "emit-recon-pack", "--run", str(run), "--config", str(config),
        "--recon-context", str(context),
    ) == 0
    execution_jobs = read_jsonl(run / "recon" / "execution_jobs.jsonl")
    execution = tmp_path / "execution"; execution.mkdir()
    receipts = []
    arxiv_index = 0; base_time = datetime.fromisoformat("2026-08-03T10:00:00+08:00")
    execution_context = json.loads((run / "recon" / "execution_context.json").read_text(encoding="utf-8"))
    for job in execution_jobs:
        if job["source"] == "ARXIV":
            paper = _arxiv_paper(
                "http://arxiv.org/abs/2501.00001v1", "Bound nearest prior",
                "https://arxiv.org/pdf/2501.00001",
            )
            raw = json.dumps({"status": "success", "results_count": 1, "papers": [paper]}).encode()
        else:
            raw = b'{"meta":{"count":0,"per_page":10},"results":[]}'
        relative = Path(str(job["raw_output_path"])); target = execution.joinpath(*relative.parts[1:])
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)
        request_time = base_time
        if job["source"] == "ARXIV":
            request_time += timedelta(seconds=3 * arxiv_index); arxiv_index += 1
        receipts.append({
            "schema_version": "idea_factory.recon_execution_receipt.v3",
            "job_id": job["job_id"], "source": job["source"], "query_id": job["query_id"],
            "argv_sha256": job["argv_sha256"], "raw_output_path": job["raw_output_path"],
            "query_pack_sha256": job["query_pack_sha256"], "command_policy_version": job["command_policy_version"],
            "started_at": request_time.isoformat(), "request_started_at": request_time.isoformat(),
            "completed_at": (request_time + timedelta(seconds=1)).isoformat(), "exit_code": 0,
            "status": "SUCCESS" if job["source"] == "ARXIV" else "EMPTY",
            "http_status": 200, "raw_file_sha256": hashlib.sha256(raw).hexdigest(),
            "execution_context_sha256": execution_context["execution_context_sha256"],
            "tool_stdout_summary": "one result" if job["source"] == "ARXIV" else "empty result",
            "tool_stderr_summary": "", "error_reason": "",
        })
    _jsonl(execution / "execution_receipts.jsonl", receipts)
    assert _invoke(
        "ingest-recon", "--run", str(run), "--config", str(config),
        "--execution-bundle", str(execution), "--allow-test-ready",
    ) == 0
    report_jobs = read_jsonl(run / "recon" / "report_jobs.jsonl")
    recon_results = tmp_path / "recon-results.jsonl"
    recon_results.write_text(
        "".join(_report_result(job, "NEAR_PRIOR_WITH_RESIDUAL", evidence=job["evidence"][0]) + "\n" for job in report_jobs),
        encoding="utf-8",
    )
    assert _invoke(
        "ingest-recon", "--run", str(run), "--config", str(config),
        "--results", str(recon_results), "--allow-test-ready",
    ) == 0

    assert _invoke("emit-route-jobs", "--run", str(run), "--config", str(config), "--allow-test-ready") == 0
    route_jobs = read_jsonl(run / "routes" / "jobs.jsonl")
    route_results = tmp_path / "route-results.jsonl"
    route_results.write_text("".join(_route_result(job) + "\n" for job in route_jobs), encoding="utf-8")
    assert _invoke(
        "ingest-routes", "--run", str(run), "--config", str(config),
        "--results", str(route_results), "--allow-test-ready",
    ) == 0
    assert _invoke("emit-review-jobs", "--run", str(run), "--config", str(config), "--allow-test-ready") == 0
    review_jobs = read_jsonl(run / "review" / "jobs.jsonl")
    decisions = ["PASS_TO_HUMAN", *("KILL" for _ in review_jobs[1:])]
    review_results = tmp_path / "review-results.jsonl"
    review_results.write_text(
        "".join(_review_result(job, decision) + "\n" for job, decision in zip(review_jobs, decisions)),
        encoding="utf-8",
    )
    assert _invoke(
        "ingest-reviews", "--run", str(run), "--config", str(config),
        "--results", str(review_results), "--allow-test-ready",
    ) == 0
    assert _invoke("ingest-human-scores", "--run", str(run), "--config", str(config), "--allow-test-ready") == 0
    human_jobs = read_jsonl(run / "ledger" / "human_scoring_jobs.jsonl")
    human_results = tmp_path / "human-results.jsonl"
    human_results.write_text("".join(_human_score(job) + "\n" for job in human_jobs), encoding="utf-8")
    assert _invoke(
        "ingest-human-scores", "--run", str(run), "--config", str(config),
        "--results", str(human_results), "--allow-test-ready",
    ) == 0
    repository = tmp_path / "repository"; repository.mkdir()
    assert _invoke(
        "finalize", "--run", str(run), "--config", str(config),
        "--repository-root", str(repository), "--allow-test-ready",
    ) == 0
    status = __import__("idea_factory.pipeline", fromlist=["status"]).status(run)
    assert status["stage"] == "COMPLETE"
    assert status["human_review_completed"] is True
    assert status["idea_pack_produced"] is True
    assert status["live_recon_executed"] is False
    assert status["real_idea_pack_produced"] is False
    assert status["cheap_test_executed"] is False
    assert (repository / "data" / "idea_ledger.jsonl").is_file()
    capsys.readouterr()
