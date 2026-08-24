from pathlib import Path

import pytest

from idea_factory.models import Opportunity, OpportunityOperator


def _build_dedup_run(
    tmp_path: Path,
    *,
    with_ready: bool = True,
    inference_flags: tuple[str, ...] = ("failure mechanism inferred",),
    legacy_row: str = "| L | other | OTHER | elsewhere | old | source |",
):
    import json
    from idea_factory.artifacts import read_jsonl
    from idea_factory.corpus import CorpusRouterConfig
    from idea_factory.legacy_ledger import import_legacy_ledger
    from idea_factory.ledger import emit_shape_assignment_jobs, ingest_shape_assignments, publish_internal_dedup
    from idea_factory.opportunities import emit_mining_jobs, ingest_opportunity_results
    from idea_factory.quality import publish_quality
    from test_opportunities import _build_run, _empty_result

    run, prompt = _build_run(tmp_path)
    mining = read_jsonl(emit_mining_jobs(run, prompt))
    results = [_empty_result(job) for job in mining]
    if with_ready:
        first = mining[0]
        wrapper = json.loads(results[0])
        wrapper["opportunities"] = [{
            "schema_version": "idea_factory.opportunity.v1", "opportunity_id": "opp-recon", "operator": first["operator"],
            "assumption_x": "The cache remains resident during long serving traces.",
            "observation_y": "Cards report retrieval degrades after eviction.", "condition_z": "under long serving traces",
            "failure_f": "cache eviction causes retrieval loss", "missing_capability_w": "eviction-aware retrieval control",
            "alternative_explanation_a": "cache eviction",
            "decisive_experiment": "Compare recall after forced cache eviction against cache eviction.",
            "supporting_card_ids": [first["card_ids"][0]], "nearest_internal_neighbors": [first["neighbor_ids"][0]],
            "scope_compatibility": "Both cards cover KV cache serving over long traces.",
            "inference_flags": list(inference_flags),
        }]
        results[0] = json.dumps(wrapper)
    ingest_opportunity_results(run / "opportunities" / "mining_jobs.jsonl", results)
    publish_quality(run)
    papers = tmp_path / "papers"; legacy = papers / "_ideas" / "idea_ledger.md"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(
        "| id | one-line | mechanism family | target | status / venue | source |\n"
        "|---|---|---|---|---|---|\n"
        f"{legacy_row}\n",
        encoding="utf-8",
    )
    config = CorpusRouterConfig(tmp_path / "notes", (papers / "list.txt",), legacy, 1, 1, ("KV_CACHE",), ())
    import_legacy_ledger(run, config)
    jobs_path = emit_shape_assignment_jobs(run, config)
    jobs = read_jsonl(jobs_path)
    shape_results = []
    for job in jobs:
        shape_results.append(json.dumps({
            "schema_version": "idea_factory.shape_assignment_result.v1",
            **{key: job[key] for key in ("job_id", "opportunity_id", "quality_record_sha256", "quality_bundle_sha256", "legacy_index_sha256")},
            "old_assumption": "cache remains resident", "failure_mechanism": "cache eviction causes retrieval loss",
            "missing_capability": "eviction-aware retrieval control", "target_scope": "KV cache serving",
            "mechanism_family": "EVICTION_CONTROL", "residual_difference": "",
        }))
    ingest_shape_assignments(jobs_path, shape_results)
    publish_internal_dedup(run, config)
    return run, config


def test_recon_public_contract_is_importable() -> None:
    from idea_factory.recon import (
        check_live_recon_preflight,
        emit_recon_query_pack,
        normalize_openalex_abstract,
        record_terms_notification,
    )

    assert callable(emit_recon_query_pack)
    assert callable(check_live_recon_preflight)
    assert normalize_openalex_abstract({"cache": [1], "state": [0]}) == "state cache"
    assert callable(record_terms_notification)


def test_query_pack_has_exactly_eight_stable_queries_per_ready_opportunity(tmp_path: Path, monkeypatch) -> None:
    from idea_factory import recon

    run = tmp_path / "run"; run.mkdir()
    opportunity = Opportunity(
        schema_version="idea_factory.opportunity.v1", opportunity_id="opp-1",
        operator=OpportunityOperator.ASSUMPTION_BREAK,
        assumption_x="Cache state stays fixed during long serving traces",
        observation_y="retrieval quality falls after eviction", condition_z="under long serving traces",
        failure_f="eviction makes retained evidence stale", missing_capability_w="version-aware invalidation",
        alternative_explanation_a="cache eviction", decisive_experiment="Compare recall and latency after forced eviction against cache eviction.",
        supporting_card_ids=["card-1"], nearest_internal_neighbors=["card-2"], scope_compatibility="same scope",
    )
    monkeypatch.setattr(recon, "_opportunity_rows", lambda _run, _config: ([{"dedup": {"status": "CLEAR", "blocked": False}, "quality": {}, "opportunity": opportunity}], {"ready": ({"status": "CLEAR"},)}))
    path = recon.emit_recon_query_pack(run, object())
    rows = [__import__("json").loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 8
    assert {(row["lane"], row["variant"]) for row in rows} == {
        (lane, variant) for lane in ("CONCEPT", "MECHANISM", "FAILURE", "EVALUATION") for variant in ("CURRENT_TERMS", "GENERIC_SHAPE")
    }
    assert len({row["query_id"] for row in rows}) == 8


def test_pure_query_generation_rejects_method_names_and_model_acronyms() -> None:
    import pytest
    from idea_factory.recon import generate_recon_queries

    opportunity = Opportunity(
        schema_version="idea_factory.opportunity.v1", opportunity_id="opp-bad", operator=OpportunityOperator.ASSUMPTION_BREAK,
        assumption_x="LoRA state remains fixed during serving", observation_y="retrieval falls after eviction",
        condition_z="under long traces", failure_f="eviction causes retrieval loss",
        missing_capability_w="version-aware invalidation", alternative_explanation_a="cache eviction",
        decisive_experiment="Compare recall against cache eviction", supporting_card_ids=["c1"],
        nearest_internal_neighbors=["c2"], scope_compatibility="same scope",
    )
    with pytest.raises(ValueError, match="method names or model acronyms"):
        generate_recon_queries(opportunity)


@pytest.mark.parametrize("term", ["Transformer", "transformer", "transformerxl", "transformer-xl", "transformer_xl", "T5", "t5", "GPT2", "gpt2", "gpt-2", "gpt_2", "BERTbase", "LLaMA", "LoRA", "QLoRA", "DPO", "PPO", "FluxNet", "ZXQ"])
def test_method_firewall_controlled_lexicon_and_style_rules(term: str) -> None:
    from idea_factory.recon import generate_recon_queries

    opportunity = Opportunity(
        schema_version="idea_factory.opportunity.v1", opportunity_id="opp-firewall", operator=OpportunityOperator.ASSUMPTION_BREAK,
        assumption_x=f"{term} state remains fixed during serving", observation_y="retrieval falls after eviction",
        condition_z="under long traces", failure_f="eviction causes retrieval loss", missing_capability_w="version-aware invalidation",
        alternative_explanation_a="cache eviction", decisive_experiment="Compare recall against cache eviction",
        supporting_card_ids=["c1"], nearest_internal_neighbors=["c2"], scope_compatibility="same scope",
    )
    with pytest.raises(ValueError, match="controlled deterministic firewall"):
        generate_recon_queries(opportunity)


def test_method_firewall_allows_explicit_kv_and_rag_domain_terms() -> None:
    from idea_factory.recon import generate_recon_queries

    opportunity = Opportunity(
        schema_version="idea_factory.opportunity.v1", opportunity_id="opp-domain", operator=OpportunityOperator.ASSUMPTION_BREAK,
        assumption_x="KV state remains fixed during serving", observation_y="RAG retrieval falls after eviction",
        condition_z="under long memory traces", failure_f="eviction causes retrieval loss", missing_capability_w="version-aware invalidation",
        alternative_explanation_a="cache eviction", decisive_experiment="Compare recall against cache eviction",
        supporting_card_ids=["c1"], nearest_internal_neighbors=["c2"], scope_compatibility="same memory scope",
    )
    assert len(generate_recon_queries(opportunity)) == 8


def test_method_firewall_stems_do_not_ban_ordinary_plural_words() -> None:
    from idea_factory.recon import generate_recon_queries

    opportunity = Opportunity(
        schema_version="idea_factory.opportunity.v1", opportunity_id="opp-ordinary", operator=OpportunityOperator.ASSUMPTION_BREAK,
        assumption_x="Llamas remain near the cache during serving", observation_y="retrieval falls after eviction",
        condition_z="under long memory traces", failure_f="eviction causes retrieval loss", missing_capability_w="version-aware invalidation",
        alternative_explanation_a="cache eviction", decisive_experiment="Compare recall against cache eviction",
        supporting_card_ids=["c1"], nearest_internal_neighbors=["c2"], scope_compatibility="same memory scope",
    )
    assert len(generate_recon_queries(opportunity)) == 8


def test_method_firewall_covers_inference_flags_before_pack_embedding() -> None:
    from idea_factory.recon import generate_recon_queries

    opportunity = Opportunity(
        schema_version="idea_factory.opportunity.v1", opportunity_id="opp-flag", operator=OpportunityOperator.ASSUMPTION_BREAK,
        assumption_x="Cache state remains fixed during serving", observation_y="retrieval falls after eviction",
        condition_z="under long traces", failure_f="eviction causes retrieval loss", missing_capability_w="version-aware invalidation",
        alternative_explanation_a="cache eviction", decisive_experiment="Compare recall against cache eviction",
        supporting_card_ids=["c1"], nearest_internal_neighbors=["c2"], scope_compatibility="same memory scope",
        inference_flags=["transformer-derived claim"],
    )
    with pytest.raises(ValueError, match="controlled deterministic firewall"):
        generate_recon_queries(opportunity)


def test_terms_notice_is_explicit_and_preflight_never_installs(tmp_path: Path) -> None:
    from idea_factory.recon import TERMS_ACKNOWLEDGEMENT, check_live_recon_preflight, record_terms_notification

    before = check_live_recon_preflight(tmp_path, {})
    assert "NOTICE_ARXIV" in before["missing"]
    paths = record_terms_notification(tmp_path, user_notified_at="2026-08-03T10:00:00+08:00", acknowledgement=TERMS_ACKNOWLEDGEMENT)
    assert "https://info.arxiv.org/help/api/index.html" in paths["arxiv"].read_text(encoding="utf-8")
    assert "https://developers.openalex.org/" in paths["openalex"].read_text(encoding="utf-8")


def _make_directory_link(link: Path, target: Path) -> None:
    import os
    import subprocess

    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        completed = subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(link), str(target)], capture_output=True)
        if completed.returncode:
            pytest.skip("directory links unavailable")


def test_terms_notice_record_rejects_licenses_junction_without_touching_outside(tmp_path: Path) -> None:
    from idea_factory.recon import TERMS_ACKNOWLEDGEMENT, record_terms_notification

    workspace = tmp_path / "workspace"; workspace.mkdir()
    outside = tmp_path / "outside"; outside.mkdir(); sentinel = outside / "sentinel.txt"; sentinel.write_text("keep", encoding="utf-8")
    _make_directory_link(workspace / ".licenses", outside)
    with pytest.raises(ValueError, match="licenses|anchored|escapes"):
        record_terms_notification(workspace, user_notified_at="2026-08-03T10:00:00+08:00", acknowledgement=TERMS_ACKNOWLEDGEMENT)
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (outside / "literature_search_arxiv_LICENSE.txt").exists()


def test_preflight_rejects_licenses_junction_without_reading_outside(tmp_path: Path) -> None:
    from idea_factory.recon import check_live_recon_preflight

    workspace = tmp_path / "workspace"; workspace.mkdir()
    outside = tmp_path / "outside"; outside.mkdir(); sentinel = outside / "literature_search_arxiv_LICENSE.txt"; sentinel.write_text("outside secret", encoding="utf-8")
    _make_directory_link(workspace / ".licenses", outside)
    result = check_live_recon_preflight(workspace, {})
    assert not result["ok"] and "LICENSES_PATH_INVALID" in result["missing"]
    assert sentinel.read_text(encoding="utf-8") == "outside secret"


def test_openalex_parser_reconstructs_abstract_and_rejects_nonfinite() -> None:
    import pytest
    from idea_factory.recon import parse_openalex_raw

    raw = '{"results":[{"id":"https://openalex.org/W1","display_name":"A paper","publication_year":2024,"doi":null,"authorships":[],"abstract_inverted_index":{"world":[1],"hello":[0]},"primary_location":{"landing_page_url":null}}]}'
    evidence = parse_openalex_raw(raw, query_id="q-1", raw_file_sha="a" * 64)
    assert evidence[0].abstract_text == "hello world"
    with pytest.raises(ValueError, match="strict JSON"):
        parse_openalex_raw('{"results": NaN}', query_id="q-1", raw_file_sha="a" * 64)


def test_raw_parsers_enforce_materialized_ten_result_caps() -> None:
    import json
    from idea_factory.recon import parse_arxiv_raw, parse_openalex_raw

    arxiv_papers = [
        _arxiv_paper(f"http://arxiv.org/abs/{index}", f"Paper {index}", f"https://arxiv.org/pdf/{index}")
        for index in range(11)
    ]
    with pytest.raises(ValueError, match="max_results"):
        parse_arxiv_raw(
            json.dumps({"status": "success", "results_count": 11, "papers": arxiv_papers}),
            query_id="q-cap", raw_file_sha="a" * 64, max_results=10,
        )
    openalex_results = [
        {
            "id": f"https://openalex.org/W{index}", "display_name": f"Paper {index}",
            "publication_year": 2024, "doi": None, "authorships": [],
            "abstract_inverted_index": None, "primary_location": None,
        }
        for index in range(11)
    ]
    with pytest.raises(ValueError, match="per_page"):
        parse_openalex_raw(
            json.dumps({"meta": {"count": 11, "per_page": 10}, "results": openalex_results}),
            query_id="q-cap", raw_file_sha="b" * 64, per_page=10,
        )
    with pytest.raises(ValueError, match="per_page"):
        parse_openalex_raw(
            json.dumps({"meta": {"count": 1, "per_page": 10.0}, "results": openalex_results[:1]}),
            query_id="q-cap", raw_file_sha="b" * 64, per_page=10,
        )


@pytest.mark.parametrize(
    "raw",
    [
        '{"results":[]}',
        '{"meta":{"count":0},"results":[]}',
        '{"meta":{"count":false,"per_page":10},"results":[]}',
        '{"meta":{"count":0,"per_page":10.0},"results":[]}',
        '{"meta":{"count":1,"per_page":10},"results":[]}',
    ],
)
def test_openalex_empty_results_require_metadata_confirmed_zero(raw: str) -> None:
    from idea_factory.recon import parse_openalex_raw

    with pytest.raises(ValueError, match="genuine zero"):
        parse_openalex_raw(raw, query_id="q-zero", raw_file_sha="a" * 64, per_page=10)
    assert parse_openalex_raw(
        '{"meta":{"count":0,"per_page":10},"results":[]}',
        query_id="q-zero", raw_file_sha="a" * 64, per_page=10,
    ) == []


def test_openalex_abstract_rejects_position_collisions_and_gaps() -> None:
    import pytest
    from idea_factory.recon import normalize_openalex_abstract

    with pytest.raises(ValueError, match="unique"):
        normalize_openalex_abstract({"first": [0], "second": [0]})
    with pytest.raises(ValueError, match="contiguous"):
        normalize_openalex_abstract({"first": [0], "third": [2]})


def _arxiv_paper(identifier: str, title: str, pdf_url: str) -> dict[str, object]:
    return {
        "entry_id": identifier,
        "title": title,
        "summary": "A useful abstract.",
        "published": "2025-01-02T00:00:00+00:00",
        "authors": ["Ada Lovelace"],
        "pdf_url": pdf_url,
        "doi": None,
    }


def test_arxiv_parser_accepts_actual_cumulative_json_stream_and_pdf_url() -> None:
    import json
    from idea_factory.recon import parse_arxiv_raw

    first = _arxiv_paper("http://arxiv.org/abs/2501.00001v1", "First", "https://arxiv.org/pdf/2501.00001")
    second = _arxiv_paper("http://arxiv.org/abs/2501.00002v1", "Second", "https://arxiv.org/pdf/2501.00002")
    raw = json.dumps({"status": "success", "results_count": 1, "papers": [first]}) + "\n" + json.dumps({"status": "success", "results_count": 2, "papers": [first, second]})
    records = parse_arxiv_raw(raw, query_id="q-1", raw_file_sha="a" * 64)
    assert [record.title for record in records] == ["First", "Second"]
    assert records[1].url == "https://arxiv.org/pdf/2501.00002"


def test_evidence_id_is_content_stable_across_query_coverage() -> None:
    import json
    from idea_factory.recon import parse_arxiv_raw

    paper = _arxiv_paper("http://arxiv.org/abs/2501.00001v1", "First", "https://arxiv.org/pdf/2501.00001")
    raw = json.dumps({"status": "success", "results_count": 1, "papers": [paper]})
    first = parse_arxiv_raw(raw, query_id="q-1", raw_file_sha="a" * 64)[0]
    second = parse_arxiv_raw(raw, query_id="q-2", raw_file_sha="b" * 64)[0]
    assert first.evidence_id == second.evidence_id


def _recon_evidence(evidence_id: str, *, source: str, source_id: str, title: str, year: int, doi: str | None, query_id: str):
    from idea_factory.recon import ReconEvidence

    return ReconEvidence(
        evidence_id=evidence_id, source=source, source_id=source_id, title=title,
        year=year, doi=doi, url=f"https://example.test/{source_id}", authors=("A",), abstract_text="abstract",
        query_id=query_id, raw_file_sha="a" * 64, source_aliases=(source_id,), query_ids=(query_id,),
    )


def test_evidence_dedup_is_transitive_and_permutation_stable() -> None:
    from itertools import permutations
    from idea_factory.recon import _dedup_evidence

    records = (
        _recon_evidence("e-a", source="ARXIV", source_id="a", title="Shared Title", year=2024, doi=None, query_id="q1"),
        _recon_evidence("e-b", source="OPENALEX", source_id="b", title=" shared   title ", year=2024, doi="10.1/x", query_id="q2"),
        _recon_evidence("e-c", source="ARXIV", source_id="c", title="Other", year=2024, doi="10.1/x", query_id="q3"),
    )
    projections = [_dedup_evidence(order) for order in permutations(records)]
    assert all(projection == projections[0] for projection in projections)
    assert len(projections[0]) == 1
    assert projections[0][0]["query_ids"] == ["q1", "q2", "q3"]
    assert projections[0][0]["source_aliases"] == ["ARXIV:a", "ARXIV:c", "OPENALEX:b"]


def test_evidence_title_dedup_does_not_merge_conflicting_dois() -> None:
    from idea_factory.recon import _dedup_evidence

    records = (
        _recon_evidence("e-x", source="ARXIV", source_id="x", title="Same", year=2024, doi="10.1/x", query_id="q1"),
        _recon_evidence("e-y", source="OPENALEX", source_id="y", title=" same ", year=2024, doi="10.1/y", query_id="q2"),
    )
    assert len(_dedup_evidence(records)) == 2


def test_doi_canonicalization_handles_prefix_url_host_case_and_unicode() -> None:
    from idea_factory.recon import _dedup_evidence

    variants = (
        "  DOI:10.123/AbC  ",
        "https://DOI.ORG/10.123/abc",
        "HTTP://www.doi.org/10.123/ABC",
        "https://DX.DOI.ORG/10.123/abc",
        "ｄｏｉ：１０．１２３／ＡＢＣ",
    )
    records = tuple(
        _recon_evidence(
            f"e-doi-{index}", source="ARXIV" if index % 2 == 0 else "OPENALEX",
            source_id=f"source-{index}", title=f"Title {index}", year=2024,
            doi=doi, query_id=f"q{index}",
        )
        for index, doi in enumerate(variants)
    )
    projection = _dedup_evidence(records)
    assert len(projection) == 1
    assert projection[0]["doi"] == "10.123/abc"


def test_arxiv_parser_rejects_nonprefix_stream_and_legacy_list() -> None:
    import json
    import pytest
    from idea_factory.recon import parse_arxiv_raw

    first = _arxiv_paper("http://arxiv.org/abs/1", "First", "https://arxiv.org/pdf/1")
    second = _arxiv_paper("http://arxiv.org/abs/2", "Second", "https://arxiv.org/pdf/2")
    stream = json.dumps({"status": "success", "results_count": 1, "papers": [first]}) + json.dumps({"status": "success", "results_count": 1, "papers": [second]})
    with pytest.raises(ValueError, match="cumulative"):
        parse_arxiv_raw(stream, query_id="q-1", raw_file_sha="a" * 64)
    with pytest.raises(ValueError, match="wrapper"):
        parse_arxiv_raw(json.dumps([first]), query_id="q-1", raw_file_sha="a" * 64)


def test_arxiv_parser_empty_output_is_not_self_declared_evidence() -> None:
    import pytest
    from idea_factory.recon import parse_arxiv_raw

    with pytest.raises(ValueError, match="empty"):
        parse_arxiv_raw("", query_id="q-1", raw_file_sha="a" * 64)


def test_evidence_parser_rejects_non_url_metadata() -> None:
    import json
    import pytest
    from idea_factory.recon import parse_arxiv_raw

    paper = _arxiv_paper("arxiv-id", "Paper", "not a URL")
    raw = json.dumps({"status": "success", "results_count": 1, "papers": [paper]})
    with pytest.raises(ValueError, match="URL"):
        parse_arxiv_raw(raw, query_id="q-1", raw_file_sha="a" * 64)


def test_synonym_variant_is_not_part_of_v1_api(tmp_path: Path) -> None:
    import pytest
    from idea_factory import recon

    run = tmp_path / "run"; run.mkdir()
    with pytest.raises(TypeError):
        recon.emit_recon_query_pack(run, object(), include_synonym=True)


def test_validate_query_pack_regenerates_from_dedup_and_rejects_tamper(tmp_path: Path) -> None:
    import json
    import pytest
    from idea_factory.recon import emit_recon_query_pack, validate_query_pack

    run, config = _build_dedup_run(tmp_path)
    path = emit_recon_query_pack(run, config)
    bundle = validate_query_pack(run, config)
    assert len(bundle.queries) == 8
    assert bundle.manifest["method_firewall_version"].startswith("idea_factory.method_model_firewall")
    assert len(bundle.manifest["method_firewall_policy_sha256"]) == 64
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["query"] = "invented acronym method"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="replay"):
        validate_query_pack(run, config)


def test_empty_query_pack_requires_genuinely_empty_validated_dedup(tmp_path: Path) -> None:
    from idea_factory.recon import emit_recon_query_pack, validate_query_pack

    run, config = _build_dedup_run(tmp_path, with_ready=False)
    emit_recon_query_pack(run, config)
    bundle = validate_query_pack(run, config)
    assert bundle.queries == ()
    assert bundle.manifest["query_count"] == 0


def test_genuinely_empty_dedup_supports_full_zero_recon_chain(tmp_path: Path) -> None:
    from idea_factory.recon import emit_recon_report_jobs, ingest_recon_reports, normalize_recon_raw, validate_recon_bundle

    run, config = _build_dedup_run(tmp_path, with_ready=False)
    _emit_empty_execution_chain(run, config)
    normalize_recon_raw(run, config)
    jobs_path = emit_recon_report_jobs(run, config)
    ingest_recon_reports(jobs_path, [], config)
    bundle = validate_recon_bundle(run, config)
    assert all(not records for records in bundle.values())


def test_execution_jobs_are_exactly_regenerated_from_validated_pack(tmp_path: Path) -> None:
    import json
    from idea_factory.recon import emit_recon_execution_job_templates, emit_recon_query_pack, materialize_execution_jobs, validate_execution_jobs

    run, config = _build_dedup_run(tmp_path)
    emit_recon_query_pack(run, config)
    templates_path = emit_recon_execution_job_templates(run, config)
    templates = [json.loads(line) for line in templates_path.read_text(encoding="utf-8").splitlines()]
    arxiv_template = next(row for row in templates if row["source"] == "ARXIV")
    assert "--max_results" in arxiv_template["args"] and "--max-results" not in arxiv_template["args"]
    assert "uv" not in arxiv_template["args"] and arxiv_template["script_key"] == "ARXIV"
    with pytest.raises(ValueError, match="execution_context"):
        materialize_execution_jobs(run, config)
    _workspace, _roots, context = _prepare_test_execution_context(run)
    jobs_path = materialize_execution_jobs(run, config)
    jobs = validate_execution_jobs(run, config)
    assert len(jobs) == 16
    for job in jobs:
        assert job["argv"][:2] == [context["uv"]["path"], "run"]
        assert job["argv"][2] == context["scripts"][job["source"]]["path"]
        assert job["cwd"] == context["scripts"][job["source"]]["skill_root"]
        assert all(token not in {">", "1>", "2>"} for token in job["argv"])
    rows = [json.loads(line) for line in jobs_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["argv"][2] = str(tmp_path / "alternate.py")
    jobs_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="replay"):
        validate_execution_jobs(run, config)


def test_tool_output_contract_is_hash_bound_across_template_context_and_materialized_job(tmp_path: Path) -> None:
    import hashlib
    import json
    from idea_factory.artifacts import read_jsonl, write_jsonl
    from idea_factory.recon import emit_recon_execution_job_templates, emit_recon_query_pack, materialize_execution_jobs, validate_execution_context, validate_execution_jobs

    expected = {
        "ARXIV": "CUMULATIVE_WRAPPERS_OR_EMPTY_STDOUT_ON_ZERO_V1",
        "OPENALEX": "OPENALEX_JSON_OBJECT_WRAPPER_V1",
    }
    run, config = _build_dedup_run(tmp_path); emit_recon_query_pack(run, config)
    templates = read_jsonl(emit_recon_execution_job_templates(run, config))
    assert {row["source"]: row["tool_output_contract"] for row in templates[:2]} == expected
    _workspace, _roots, context = _prepare_test_execution_context(run)
    assert context["tool_output_contracts"] == expected
    jobs_path = materialize_execution_jobs(run, config); jobs = read_jsonl(jobs_path)
    assert all(job["tool_output_contract"] == expected[job["source"]] for job in jobs)
    jobs[0]["tool_output_contract"] = "UNBOUND_OUTPUT"
    write_jsonl(jobs_path, jobs)
    with pytest.raises(ValueError, match="replay"):
        validate_execution_jobs(run, config)
    materialize_execution_jobs(run, config)
    context_path = run / "recon" / "execution_context.json"
    tampered = json.loads(context_path.read_text(encoding="utf-8")); tampered["tool_output_contracts"]["ARXIV"] = "UNBOUND_OUTPUT"
    body = {key: value for key, value in tampered.items() if key != "execution_context_sha256"}
    tampered["execution_context_sha256"] = hashlib.sha256(json.dumps(body, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    context_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="tool output contract"):
        validate_execution_context(run)


def _fake_uv(root: Path, version: str = "uv 0.9.1") -> Path:
    import hashlib
    import os
    import shutil
    import subprocess
    import tempfile

    if os.name == "nt":
        cache = Path(tempfile.gettempdir()) / "idea-factory-native-uv-fixtures" / hashlib.sha256(version.encode()).hexdigest()
        executable = cache / "uv.exe"
        if not executable.is_file():
            cache.mkdir(parents=True, exist_ok=True)
            source = cache / "Program.cs"
            escaped = version.replace("\\", "\\\\").replace('"', '\\"')
            source.write_text(f'using System; class Program {{ static void Main() {{ Console.WriteLine("{escaped}"); }} }}', encoding="utf-8")
            compiler = Path(os.environ["WINDIR"]) / "Microsoft.NET" / "Framework64" / "v4.0.30319" / "csc.exe"
            subprocess.run([str(compiler), "/nologo", "/target:exe", f"/out:{executable}", str(source)], check=True, capture_output=True)
        path = root / "uv.exe"; shutil.copy2(executable, path)
    else:
        path = root / "uv"; path.write_text(f"#!/bin/sh\necho '{version}'\n", encoding="utf-8"); path.chmod(0o755)
    return path


def _fake_skill_roots(workspace: Path, *, prefix: str = "fixture") -> dict[str, Path]:
    roots = {}
    for key, script_name in (("arxiv", "search_arxiv.py"), ("openalex", "openalex_cli.py")):
        root = workspace / f"{prefix}-{key}"; script = root / "scripts" / script_name
        script.parent.mkdir(parents=True, exist_ok=True); script.write_text("# fake reviewed skill script\n", encoding="utf-8")
        roots[key] = root
    return roots


def test_live_preflight_accepts_official_uv_build_metadata(tmp_path: Path) -> None:
    from idea_factory.recon import TERMS_ACKNOWLEDGEMENT, check_live_recon_preflight, record_terms_notification

    record_terms_notification(
        tmp_path,
        user_notified_at="2026-08-03T09:00:00+08:00",
        acknowledgement=TERMS_ACKNOWLEDGEMENT,
    )
    result = check_live_recon_preflight(
        tmp_path,
        _fake_skill_roots(tmp_path),
        uv_executable=_fake_uv(tmp_path, "uv 0.12.1 (329541a50 2026-07-31 x86_64-pc-windows-msvc)"),
    )

    assert result["ok"] is True
    assert result["uv_version"] == "uv 0.12.1 (329541a50 2026-07-31 x86_64-pc-windows-msvc)"


@pytest.mark.parametrize(
    "version",
    [
        "uv 0.12.1 (not an official build identity)",
        "uv 0.12.1 (329541a50 2026-99-99 x86_64-pc-windows-msvc)",
        "uv 0.12.1 (329541a50 2026-07-31 target)",
    ],
)
def test_live_preflight_rejects_malformed_uv_build_metadata(tmp_path: Path, version: str) -> None:
    from idea_factory.recon import TERMS_ACKNOWLEDGEMENT, check_live_recon_preflight, record_terms_notification

    record_terms_notification(
        tmp_path,
        user_notified_at="2026-08-03T09:00:00+08:00",
        acknowledgement=TERMS_ACKNOWLEDGEMENT,
    )
    result = check_live_recon_preflight(
        tmp_path,
        _fake_skill_roots(tmp_path),
        uv_executable=_fake_uv(tmp_path, version),
    )

    assert result["ok"] is False
    assert "UV_VERSION_UNAVAILABLE" in result["missing"]


def _operator_attestation_body(workspace: Path, roots: dict[str, Path], uv: Path, *, scope: str = "TEST_ONLY") -> dict[str, object]:
    from idea_factory.artifacts import sha256_file
    from idea_factory.recon import _directory_manifest_sha256

    scripts = {}
    for source, key, script_name, package_name in (
        ("ARXIV", "arxiv", "search_arxiv.py", "literature_search_arxiv"),
        ("OPENALEX", "openalex", "openalex_cli.py", "literature_search_openalex"),
    ):
        root = roots[key].resolve(); script = (root / "scripts" / script_name).resolve()
        scripts[source] = {
            "package_name": package_name, "skill_root": str(root),
            "skill_root_sha256": _directory_manifest_sha256(root),
            "path": str(script), "sha256": sha256_file(script),
        }
    return {
        "schema_version": "idea_factory.operator_trust_attestation.v1",
        "record_kind": "OPERATOR_TRUST_ATTESTATION", "attestation_scope": scope,
        "authenticity": "NOT_AUTHENTICATED",
        "boundary": "TEST_ONLY/NOT_AUTHENTICATED" if scope == "TEST_ONLY" else "LIVE_OPERATOR_ATTESTED/NOT_AUTHENTICATED",
        "allowed_use": "IDEA_FACTORY_TASK9_RECON_EXECUTION",
        "acknowledged_at": "2026-08-03T09:04:00+08:00",
        "workspace_root": str(workspace.resolve()),
        "uv": {"path": str(uv.resolve()), "sha256": sha256_file(uv), "executable_policy": "NATIVE_UV_EXECUTABLE"},
        "scripts": scripts,
    }


def _prepare_test_execution_context(run: Path):
    from idea_factory.recon import TERMS_ACKNOWLEDGEMENT, prepare_execution_context, record_operator_trust_attestation, record_terms_notification

    workspace = run.parent
    record_terms_notification(workspace, user_notified_at="2026-08-03T09:00:00+08:00", acknowledgement=TERMS_ACKNOWLEDGEMENT)
    roots = _fake_skill_roots(workspace)
    uv = _fake_uv(workspace)
    record_operator_trust_attestation(run, _operator_attestation_body(workspace, roots, uv))
    context = prepare_execution_context(
        run, workspace_root=workspace, skill_roots=roots,
        prepared_at="2026-08-03T09:05:00+08:00", uv_executable=uv,
        allow_test_attestation=True,
    )
    return workspace, roots, context


def test_prepare_execution_context_requires_independent_operator_attestation(tmp_path: Path) -> None:
    from idea_factory.recon import TERMS_ACKNOWLEDGEMENT, prepare_execution_context, record_terms_notification

    run = tmp_path / "run"; run.mkdir(); (run / "recon").mkdir()
    record_terms_notification(tmp_path, user_notified_at="2026-08-03T09:00:00+08:00", acknowledgement=TERMS_ACKNOWLEDGEMENT)
    roots = _fake_skill_roots(tmp_path); uv = _fake_uv(tmp_path)
    with pytest.raises(ValueError, match="operator trust attestation"):
        prepare_execution_context(
            run, workspace_root=tmp_path, skill_roots=roots,
            prepared_at="2026-08-03T09:05:00+08:00", uv_executable=uv,
        )


def test_test_only_attestation_requires_explicit_opt_in_and_binds_context_and_jobs(tmp_path: Path) -> None:
    from idea_factory.artifacts import read_jsonl
    from idea_factory.recon import (
        TERMS_ACKNOWLEDGEMENT, emit_recon_execution_job_templates, emit_recon_query_pack,
        materialize_execution_jobs, prepare_execution_context,
        record_operator_trust_attestation, record_terms_notification,
    )

    run, config = _build_dedup_run(tmp_path); emit_recon_query_pack(run, config); emit_recon_execution_job_templates(run, config)
    record_terms_notification(tmp_path, user_notified_at="2026-08-03T09:00:00+08:00", acknowledgement=TERMS_ACKNOWLEDGEMENT)
    roots = _fake_skill_roots(tmp_path); uv = _fake_uv(tmp_path)
    attestation = record_operator_trust_attestation(run, _operator_attestation_body(tmp_path, roots, uv))
    with pytest.raises(ValueError, match="TEST_ONLY"):
        prepare_execution_context(
            run, workspace_root=tmp_path, skill_roots=roots,
            prepared_at="2026-08-03T09:05:00+08:00", uv_executable=uv,
        )
    context = prepare_execution_context(
        run, workspace_root=tmp_path, skill_roots=roots,
        prepared_at="2026-08-03T09:05:00+08:00", uv_executable=uv,
        allow_test_attestation=True,
    )
    assert context["operator_trust_attestation_sha256"] == attestation["operator_trust_attestation_sha256"]
    assert context["operator_trust_scope"] == "TEST_ONLY"
    jobs = read_jsonl(materialize_execution_jobs(run, config))
    assert all(job["operator_trust_attestation_sha256"] == attestation["operator_trust_attestation_sha256"] for job in jobs)


def test_attestation_rejects_drift_and_live_scope_requires_installed_package_names(tmp_path: Path) -> None:
    from idea_factory.recon import record_operator_trust_attestation, validate_operator_trust_attestation

    run = tmp_path / "run"; run.mkdir(); (run / "recon").mkdir()
    roots = _fake_skill_roots(tmp_path); uv = _fake_uv(tmp_path)
    with pytest.raises(ValueError, match="installed package names"):
        record_operator_trust_attestation(run, _operator_attestation_body(tmp_path, roots, uv, scope="LIVE_OPERATOR_ATTESTED"))
    record_operator_trust_attestation(run, _operator_attestation_body(tmp_path, roots, uv))
    (roots["arxiv"] / "scripts" / "search_arxiv.py").write_text("# drifted\n", encoding="utf-8")
    with pytest.raises(ValueError, match="attestation.*mismatch|root hash"):
        validate_operator_trust_attestation(run, allow_test_attestation=True)


def test_attestation_acknowledgement_must_precede_context_preparation(tmp_path: Path) -> None:
    from idea_factory.recon import TERMS_ACKNOWLEDGEMENT, prepare_execution_context, record_operator_trust_attestation, record_terms_notification

    run = tmp_path / "run"; run.mkdir(); (run / "recon").mkdir()
    record_terms_notification(tmp_path, user_notified_at="2026-08-03T09:00:00+08:00", acknowledgement=TERMS_ACKNOWLEDGEMENT)
    roots = _fake_skill_roots(tmp_path); uv = _fake_uv(tmp_path)
    body = _operator_attestation_body(tmp_path, roots, uv); body["acknowledged_at"] = "2026-08-03T09:06:00+08:00"
    record_operator_trust_attestation(run, body)
    with pytest.raises(ValueError, match="acknowledged_at.*prepared_at"):
        prepare_execution_context(
            run, workspace_root=tmp_path, skill_roots=roots,
            prepared_at="2026-08-03T09:05:00+08:00", uv_executable=uv,
            allow_test_attestation=True,
        )


def test_execution_context_requires_successful_preflight(tmp_path: Path) -> None:
    import sys
    from idea_factory.recon import prepare_execution_context

    run = tmp_path / "run"; run.mkdir(); (run / "recon").mkdir()
    with pytest.raises(ValueError, match="preflight"):
        prepare_execution_context(run, workspace_root=tmp_path, skill_roots={}, prepared_at="2026-08-03T09:05:00+08:00", uv_executable=Path(sys.executable))
    assert not (run / "recon" / "execution_context.json").exists()


def test_preflight_rejects_python_executable_masquerading_as_uv(tmp_path: Path) -> None:
    import sys
    from idea_factory.recon import TERMS_ACKNOWLEDGEMENT, check_live_recon_preflight, record_terms_notification

    record_terms_notification(tmp_path, user_notified_at="2026-08-03T09:00:00+08:00", acknowledgement=TERMS_ACKNOWLEDGEMENT)
    roots = {}
    for key, script_name in (("arxiv", "search_arxiv.py"), ("openalex", "openalex_cli.py")):
        root = tmp_path / key; script = root / "scripts" / script_name; script.parent.mkdir(parents=True); script.write_text("# fake\n", encoding="utf-8"); roots[key] = root
    result = check_live_recon_preflight(tmp_path, roots, uv_executable=Path(sys.executable))
    assert not result["ok"]
    assert "UV_IDENTITY_INVALID" in result["missing"]


def test_preflight_rejects_uv_named_wrapper_with_non_uv_version(tmp_path: Path) -> None:
    from idea_factory.recon import TERMS_ACKNOWLEDGEMENT, check_live_recon_preflight, record_terms_notification

    record_terms_notification(tmp_path, user_notified_at="2026-08-03T09:00:00+08:00", acknowledgement=TERMS_ACKNOWLEDGEMENT)
    roots = {}
    for key, script_name in (("arxiv", "search_arxiv.py"), ("openalex", "openalex_cli.py")):
        root = tmp_path / key; script = root / "scripts" / script_name; script.parent.mkdir(parents=True); script.write_text("# fake\n", encoding="utf-8"); roots[key] = root
    result = check_live_recon_preflight(tmp_path, roots, uv_executable=_fake_uv(tmp_path, "Python 3.12.0"))
    assert not result["ok"] and "UV_VERSION_UNAVAILABLE" in result["missing"]


@pytest.mark.skipif(__import__("os").name != "nt", reason="Windows wrapper policy")
def test_preflight_rejects_uv_cmd_wrapper_even_with_valid_version(tmp_path: Path) -> None:
    from idea_factory.recon import TERMS_ACKNOWLEDGEMENT, check_live_recon_preflight, record_terms_notification

    record_terms_notification(tmp_path, user_notified_at="2026-08-03T09:00:00+08:00", acknowledgement=TERMS_ACKNOWLEDGEMENT)
    roots = {}
    for key, script_name in (("arxiv", "search_arxiv.py"), ("openalex", "openalex_cli.py")):
        root = tmp_path / key; script = root / "scripts" / script_name
        script.parent.mkdir(parents=True); script.write_text("# fake\n", encoding="utf-8"); roots[key] = root
    wrapper = tmp_path / "uv.cmd"; wrapper.write_text("@echo off\r\necho uv 0.9.1\r\n", encoding="utf-8")
    result = check_live_recon_preflight(tmp_path, roots, uv_executable=wrapper)
    assert not result["ok"] and "UV_IDENTITY_INVALID" in result["missing"]


def test_execution_context_rejects_notice_with_naive_timestamp(tmp_path: Path) -> None:
    from idea_factory.recon import TERMS_ACKNOWLEDGEMENT, prepare_execution_context, record_terms_notification

    run = tmp_path / "run"; run.mkdir(); (run / "recon").mkdir()
    paths = record_terms_notification(tmp_path, user_notified_at="2026-08-03T09:00:00+08:00", acknowledgement=TERMS_ACKNOWLEDGEMENT)
    paths["arxiv"].write_text(paths["arxiv"].read_text(encoding="utf-8").replace("+08:00", ""), encoding="utf-8")
    roots = {}
    for key, script_name in (("arxiv", "search_arxiv.py"), ("openalex", "openalex_cli.py")):
        root = tmp_path / key; script = root / "scripts" / script_name; script.parent.mkdir(parents=True); script.write_text("# fake\n", encoding="utf-8"); roots[key] = root
    with pytest.raises(ValueError, match="preflight"):
        prepare_execution_context(run, workspace_root=tmp_path, skill_roots=roots, prepared_at="2026-08-03T09:05:00+08:00", uv_executable=_fake_uv(tmp_path))


def test_execution_context_rejects_notification_after_preparation(tmp_path: Path) -> None:
    from idea_factory.recon import TERMS_ACKNOWLEDGEMENT, prepare_execution_context, record_terms_notification

    run = tmp_path / "run"; run.mkdir(); (run / "recon").mkdir()
    record_terms_notification(tmp_path, user_notified_at="2026-08-03T09:06:00+08:00", acknowledgement=TERMS_ACKNOWLEDGEMENT)
    roots = {}
    for key, script_name in (("arxiv", "search_arxiv.py"), ("openalex", "openalex_cli.py")):
        root = tmp_path / key; script = root / "scripts" / script_name
        script.parent.mkdir(parents=True); script.write_text("# fake\n", encoding="utf-8"); roots[key] = root
    with pytest.raises(ValueError, match="notified_at.*prepared_at"):
        prepare_execution_context(
            run, workspace_root=tmp_path, skill_roots=roots,
            prepared_at="2026-08-03T09:05:00+08:00", uv_executable=_fake_uv(tmp_path),
        )


def _emit_empty_execution_chain(run: Path, config):
    import hashlib
    from datetime import datetime, timedelta
    from idea_factory.artifacts import read_jsonl, write_jsonl
    from idea_factory.recon import emit_recon_execution_job_templates, emit_recon_query_pack, materialize_execution_jobs

    emit_recon_query_pack(run, config)
    templates = read_jsonl(emit_recon_execution_job_templates(run, config))
    if templates:
        context = _prepare_test_execution_context(run)
        jobs = read_jsonl(materialize_execution_jobs(run, config))
    else:
        context = (None, None, {"execution_context_sha256": None}); jobs = []
    receipts = []
    arxiv_request_index = 0
    base_request_time = datetime.fromisoformat("2026-08-03T10:00:00+08:00")
    for job in jobs:
        raw = run / job["raw_output_path"]
        raw.parent.mkdir(parents=True, exist_ok=True)
        body = b"" if job["source"] == "ARXIV" else b'{"meta":{"count":0,"per_page":10},"results":[]}'
        raw.write_bytes(body)
        request_time = base_request_time
        if job["source"] == "ARXIV":
            request_time += timedelta(seconds=3 * arxiv_request_index)
            arxiv_request_index += 1
        receipts.append({
            "schema_version": "idea_factory.recon_execution_receipt.v3", "job_id": job["job_id"],
            "source": job["source"], "query_id": job["query_id"], "argv_sha256": job["argv_sha256"],
            "raw_output_path": job["raw_output_path"], "query_pack_sha256": job["query_pack_sha256"],
            "command_policy_version": job["command_policy_version"], "started_at": request_time.isoformat(),
            "request_started_at": request_time.isoformat(), "completed_at": (request_time + timedelta(seconds=1)).isoformat(),
            "exit_code": 0, "status": "EMPTY", "http_status": 200,
            "raw_file_sha256": hashlib.sha256(body).hexdigest(),
            "execution_context_sha256": context[2]["execution_context_sha256"],
            "tool_stdout_summary": "empty result", "tool_stderr_summary": "", "error_reason": "",
        })
    write_jsonl(run / "recon" / "execution_receipts.jsonl", receipts)
    return jobs, receipts


def test_receipts_recompute_current_notice_and_script_hashes(tmp_path: Path) -> None:
    from idea_factory.recon import validate_execution_receipts

    run, config = _build_dedup_run(tmp_path)
    _jobs, _receipts = _emit_empty_execution_chain(run, config)
    assert validate_execution_receipts(run, config)
    notice = tmp_path / ".licenses" / "literature_search_arxiv_LICENSE.txt"
    notice.write_text(notice.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="notice hash"):
        validate_execution_receipts(run, config)


def test_nonempty_receipts_require_context_and_reject_free_notice_hashes(tmp_path: Path) -> None:
    from idea_factory.artifacts import write_jsonl
    from idea_factory.recon import validate_execution_receipts

    run, config = _build_dedup_run(tmp_path)
    _jobs, receipts = _emit_empty_execution_chain(run, config)
    (run / "recon" / "execution_context.json").unlink()
    with pytest.raises(ValueError, match="execution_context"):
        validate_execution_receipts(run, config)
    _prepare_test_execution_context(run)
    receipts[0]["terms_notice_hashes"] = {"ARXIV": "a" * 64, "OPENALEX": "b" * 64}
    write_jsonl(run / "recon" / "execution_receipts.jsonl", receipts)
    with pytest.raises(ValueError, match="schema is not exact"):
        validate_execution_receipts(run, config)


def test_receipts_reject_missing_current_notice_and_expected_root_mismatch(tmp_path: Path) -> None:
    from idea_factory.recon import validate_execution_receipts

    run, config = _build_dedup_run(tmp_path)
    _emit_empty_execution_chain(run, config)
    with pytest.raises(ValueError, match="expected replay root"):
        validate_execution_receipts(run, config, expected_workspace_root=tmp_path / "elsewhere")
    (tmp_path / ".licenses" / "literature_search_openalex_LICENSE.txt").unlink()
    with pytest.raises(ValueError, match="notices are missing"):
        validate_execution_receipts(run, config)


def test_receipts_reject_tampered_execution_context_and_script(tmp_path: Path) -> None:
    import json
    from idea_factory.recon import validate_execution_receipts

    run, config = _build_dedup_run(tmp_path)
    _jobs, _receipts = _emit_empty_execution_chain(run, config)
    context_path = run / "recon" / "execution_context.json"
    context = json.loads(context_path.read_text(encoding="utf-8"))
    script = Path(context["scripts"]["ARXIV"]["path"])
    script.write_text(script.read_text(encoding="utf-8") + "# tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="script hash"):
        validate_execution_receipts(run, config)
    script.write_text("# fake reviewed skill script\n", encoding="utf-8")
    context["prepared_at"] = "2026-08-03T09:06:00+08:00"
    context_path.write_text(json.dumps(context), encoding="utf-8")
    with pytest.raises(ValueError, match="context hash"):
        validate_execution_receipts(run, config)


def test_receipts_rehash_uv_executable(tmp_path: Path) -> None:
    import json
    from idea_factory.recon import validate_execution_receipts

    run, config = _build_dedup_run(tmp_path); _emit_empty_execution_chain(run, config)
    context = json.loads((run / "recon" / "execution_context.json").read_text(encoding="utf-8"))
    uv = Path(context["uv"]["path"]); uv.write_bytes(uv.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="uv executable hash"):
        validate_execution_receipts(run, config)


def test_unsigned_context_declares_no_authenticity_and_self_consistent_rewrite_is_outside_claim(tmp_path: Path) -> None:
    import hashlib
    import json
    from idea_factory.artifacts import read_jsonl, write_jsonl
    from idea_factory.recon import materialize_execution_jobs, validate_execution_receipts

    run, config = _build_dedup_run(tmp_path); _emit_empty_execution_chain(run, config)
    context_path = run / "recon" / "execution_context.json"
    context = json.loads(context_path.read_text(encoding="utf-8"))
    assert context["record_kind"] == "UNSIGNED_NOTIFICATION_INTEGRITY"
    assert context["authenticity"] == "NOT_AUTHENTICATED"
    assert "not authenticated" in context["claim_boundary"].lower()
    context["prepared_at"] = "2026-08-03T09:06:00+08:00"
    body = {key: value for key, value in context.items() if key != "execution_context_sha256"}
    context["execution_context_sha256"] = hashlib.sha256(json.dumps(body, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    context_path.write_text(json.dumps(context), encoding="utf-8")
    receipts = read_jsonl(run / "recon" / "execution_receipts.jsonl")
    by_query_source = {(row["query_id"], row["source"]): row for row in receipts}
    jobs = read_jsonl(materialize_execution_jobs(run, config))
    rewritten = []
    for job in jobs:
        receipt = by_query_source[(job["query_id"], job["source"])]
        receipt.update(
            job_id=job["job_id"],
            argv_sha256=job["argv_sha256"],
            execution_context_sha256=context["execution_context_sha256"],
        )
        rewritten.append(receipt)
    write_jsonl(run / "recon" / "execution_receipts.jsonl", rewritten)
    assert validate_execution_receipts(run, config)


def test_receipts_bind_exact_job_hash_time_and_current_raw(tmp_path: Path) -> None:
    import json
    import pytest
    from idea_factory.artifacts import write_jsonl
    from idea_factory.recon import validate_execution_receipts

    run, config = _build_dedup_run(tmp_path)
    jobs, receipts = _emit_empty_execution_chain(run, config)
    assert len(validate_execution_receipts(run, config)) == len(jobs)
    receipts[0]["completed_at"] = "2026-08-03T09:59:59+08:00"
    write_jsonl(run / "recon" / "execution_receipts.jsonl", receipts)
    with pytest.raises(ValueError, match="before"):
        validate_execution_receipts(run, config)


def test_receipts_enforce_cross_arxiv_request_spacing(tmp_path: Path) -> None:
    from datetime import datetime, timedelta
    from idea_factory.artifacts import write_jsonl
    from idea_factory.recon import validate_execution_receipts

    run, config = _build_dedup_run(tmp_path); _jobs, receipts = _emit_empty_execution_chain(run, config)
    arxiv = [row for row in receipts if row["source"] == "ARXIV"]
    for index, receipt in enumerate(arxiv):
        receipt["request_started_at"] = f"2026-08-03T10:00:{index * 3:02d}+08:00"
    arxiv[1]["request_started_at"] = "2026-08-03T10:00:02+08:00"
    for receipt in arxiv:
        request_time = datetime.fromisoformat(receipt["request_started_at"])
        receipt["started_at"] = request_time.isoformat()
        receipt["completed_at"] = (request_time + timedelta(seconds=1)).isoformat()
    for receipt in receipts:
        receipt.setdefault("request_started_at", receipt["started_at"])
    write_jsonl(run / "recon" / "execution_receipts.jsonl", receipts)
    with pytest.raises(ValueError, match="arXiv request spacing"):
        validate_execution_receipts(run, config)


def test_receipts_reject_execution_before_context_preparation(tmp_path: Path) -> None:
    from idea_factory.artifacts import write_jsonl
    from idea_factory.recon import validate_execution_receipts

    run, config = _build_dedup_run(tmp_path); _jobs, receipts = _emit_empty_execution_chain(run, config)
    for receipt in receipts:
        receipt["request_started_at"] = receipt["started_at"]
    receipts[0]["started_at"] = "2026-08-03T09:04:59+08:00"
    receipts[0]["request_started_at"] = receipts[0]["started_at"]
    write_jsonl(run / "recon" / "execution_receipts.jsonl", receipts)
    with pytest.raises(ValueError, match="prepared_at.*execution_started_at"):
        validate_execution_receipts(run, config)


def test_receipts_reject_noncanonical_jsonl_order(tmp_path: Path) -> None:
    from idea_factory.artifacts import write_jsonl
    from idea_factory.recon import validate_execution_receipts

    run, config = _build_dedup_run(tmp_path); _jobs, receipts = _emit_empty_execution_chain(run, config)
    write_jsonl(run / "recon" / "execution_receipts.jsonl", list(reversed(receipts)))
    with pytest.raises(ValueError, match="canonical job order"):
        validate_execution_receipts(run, config)


def test_receipt_exit_code_must_match_success_status(tmp_path: Path) -> None:
    from idea_factory.artifacts import write_jsonl
    from idea_factory.recon import validate_execution_receipts

    run, config = _build_dedup_run(tmp_path); _jobs, receipts = _emit_empty_execution_chain(run, config)
    for receipt in receipts:
        receipt["exit_code"] = 0
    receipts[0]["exit_code"] = 1
    write_jsonl(run / "recon" / "execution_receipts.jsonl", receipts)
    with pytest.raises(ValueError, match="exit_code.*status"):
        validate_execution_receipts(run, config)


def test_receipt_rejects_stale_raw_hash(tmp_path: Path) -> None:
    import pytest
    from idea_factory.recon import validate_execution_receipts

    run, config = _build_dedup_run(tmp_path)
    jobs, _receipts = _emit_empty_execution_chain(run, config)
    (run / jobs[0]["raw_output_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="raw hash"):
        validate_execution_receipts(run, config)


def test_openalex_401_or_429_requires_credentials_protocol_status(tmp_path: Path) -> None:
    import pytest
    from idea_factory.artifacts import write_jsonl
    from idea_factory.recon import validate_execution_receipts

    run, config = _build_dedup_run(tmp_path)
    _jobs, receipts = _emit_empty_execution_chain(run, config)
    target = next(row for row in receipts if row["source"] == "OPENALEX")
    target["http_status"] = 429
    target["status"] = "SUCCESS"
    target["error_reason"] = ""
    write_jsonl(run / "recon" / "execution_receipts.jsonl", receipts)
    with pytest.raises(ValueError, match="CREDENTIALS_PROTOCOL_REQUIRED"):
        validate_execution_receipts(run, config)


def test_normalized_bundle_replays_from_current_raw_and_rejects_manifest_tamper(tmp_path: Path) -> None:
    import json
    import pytest
    from idea_factory.recon import normalize_recon_raw, validate_normalized_evidence_bundle

    run, config = _build_dedup_run(tmp_path)
    _emit_empty_execution_chain(run, config)
    manifest_path = normalize_recon_raw(run, config)
    assert validate_normalized_evidence_bundle(run, config).manifest["evidence_count"] == 0
    body = json.loads(manifest_path.read_text(encoding="utf-8")); body["evidence_count"] = 1
    manifest_path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="normalized evidence replay"):
        validate_normalized_evidence_bundle(run, config)


def test_official_arxiv_empty_stdout_is_genuine_zero_only_with_valid_empty_receipt(tmp_path: Path) -> None:
    import hashlib
    from idea_factory.artifacts import write_jsonl
    from idea_factory.recon import normalize_recon_raw, validate_normalized_evidence_bundle

    run, config = _build_dedup_run(tmp_path); jobs, receipts = _emit_empty_execution_chain(run, config)
    target = next(job for job in jobs if job["source"] == "ARXIV")
    (run / target["raw_output_path"]).write_bytes(b"")
    receipt = next(row for row in receipts if row["job_id"] == target["job_id"])
    receipt["status"] = "EMPTY"; receipt["raw_file_sha256"] = hashlib.sha256(b"").hexdigest()
    write_jsonl(run / "recon" / "execution_receipts.jsonl", receipts)
    normalize_recon_raw(run, config)
    assert validate_normalized_evidence_bundle(run, config).evidence == ()


def test_arxiv_empty_stdout_rejects_success_receipt(tmp_path: Path) -> None:
    import hashlib
    from idea_factory.artifacts import write_jsonl
    from idea_factory.recon import normalize_recon_raw

    run, config = _build_dedup_run(tmp_path); jobs, receipts = _emit_empty_execution_chain(run, config)
    target = next(job for job in jobs if job["source"] == "ARXIV")
    (run / target["raw_output_path"]).write_bytes(b"")
    receipt = next(row for row in receipts if row["job_id"] == target["job_id"])
    receipt["status"] = "SUCCESS"; receipt["exit_code"] = 0
    receipt["raw_file_sha256"] = hashlib.sha256(b"").hexdigest()
    write_jsonl(run / "recon" / "execution_receipts.jsonl", receipts)
    with pytest.raises(ValueError, match="empty"):
        normalize_recon_raw(run, config)


def test_credentials_protocol_receipt_skips_nonempty_raw_parsing(tmp_path: Path) -> None:
    from idea_factory.artifacts import write_jsonl
    from idea_factory.recon import normalize_recon_raw, validate_normalized_evidence_bundle

    run, config = _build_dedup_run(tmp_path); jobs, receipts = _emit_empty_execution_chain(run, config)
    target = next(job for job in jobs if job["source"] == "OPENALEX")
    (run / target["raw_output_path"]).write_bytes(b"credential diagnostic, not JSON")
    receipt = next(row for row in receipts if row["job_id"] == target["job_id"])
    receipt["status"] = "CREDENTIALS_PROTOCOL_REQUIRED"; receipt["exit_code"] = 1
    receipt["http_status"] = 429; receipt["raw_file_sha256"] = None
    receipt["error_reason"] = "credentials protocol required"
    write_jsonl(run / "recon" / "execution_receipts.jsonl", receipts)
    normalize_recon_raw(run, config)
    assert validate_normalized_evidence_bundle(run, config).evidence == ()


def test_arxiv_synthetic_zero_wrapper_is_rejected_by_official_tool_contract(tmp_path: Path) -> None:
    import hashlib
    from idea_factory.artifacts import write_jsonl
    from idea_factory.recon import normalize_recon_raw

    run, config = _build_dedup_run(tmp_path); jobs, receipts = _emit_empty_execution_chain(run, config)
    target = next(job for job in jobs if job["source"] == "ARXIV")
    body = b'{"status":"success","results_count":0,"papers":[]}'
    (run / target["raw_output_path"]).write_bytes(body)
    receipt = next(row for row in receipts if row["job_id"] == target["job_id"])
    receipt["raw_file_sha256"] = hashlib.sha256(body).hexdigest()
    write_jsonl(run / "recon" / "execution_receipts.jsonl", receipts)
    with pytest.raises(ValueError, match="synthetic zero wrapper|empty stdout"):
        normalize_recon_raw(run, config)


def test_normalization_removes_stale_per_query_files_on_rerun(tmp_path: Path) -> None:
    from idea_factory.recon import normalize_recon_raw, validate_normalized_evidence_bundle

    run, config = _build_dedup_run(tmp_path)
    _emit_empty_execution_chain(run, config)
    stale = run / "recon" / "normalized" / "arxiv" / "stale.jsonl"
    stale.parent.mkdir(parents=True, exist_ok=True); stale.write_text('{"stale":true}\n', encoding="utf-8")
    normalize_recon_raw(run, config)
    assert not stale.exists()
    assert validate_normalized_evidence_bundle(run, config).evidence == ()


def test_normalization_uses_actual_arxiv_stream_and_preserves_query_coverage(tmp_path: Path) -> None:
    import hashlib
    import json
    from idea_factory.artifacts import write_jsonl
    from idea_factory.recon import normalize_recon_raw, validate_normalized_evidence_bundle

    run, config = _build_dedup_run(tmp_path)
    jobs, receipts = _emit_empty_execution_chain(run, config)
    target_job = next(job for job in jobs if job["source"] == "ARXIV")
    paper = _arxiv_paper("http://arxiv.org/abs/2501.00001v1", "First", "https://arxiv.org/pdf/2501.00001")
    body = json.dumps({"status": "success", "results_count": 1, "papers": [paper]}).encode()
    (run / target_job["raw_output_path"]).write_bytes(body)
    receipt = next(row for row in receipts if row["job_id"] == target_job["job_id"])
    receipt["status"] = "SUCCESS"; receipt["raw_file_sha256"] = hashlib.sha256(body).hexdigest(); receipt["tool_stdout_summary"] = "one result"
    write_jsonl(run / "recon" / "execution_receipts.jsonl", receipts)
    normalize_recon_raw(run, config)
    evidence = validate_normalized_evidence_bundle(run, config).evidence
    assert len(evidence) == 1
    assert evidence[0]["url"] == "https://arxiv.org/pdf/2501.00001"
    assert target_job["query_id"] in evidence[0]["query_ids"]


def _emit_normalized_chain(run: Path, config, *, with_evidence: bool):
    import hashlib
    import json
    from idea_factory.artifacts import write_jsonl
    from idea_factory.recon import normalize_recon_raw

    jobs, receipts = _emit_empty_execution_chain(run, config)
    if with_evidence:
        target_job = next(job for job in jobs if job["source"] == "ARXIV")
        paper = _arxiv_paper("http://arxiv.org/abs/2501.00001v1", "First prior", "https://arxiv.org/pdf/2501.00001")
        body = json.dumps({"status": "success", "results_count": 1, "papers": [paper]}).encode()
        (run / target_job["raw_output_path"]).write_bytes(body)
        receipt = next(row for row in receipts if row["job_id"] == target_job["job_id"])
        receipt["status"] = "SUCCESS"; receipt["raw_file_sha256"] = hashlib.sha256(body).hexdigest()
        receipt["tool_stdout_summary"] = "one result"
        write_jsonl(run / "recon" / "execution_receipts.jsonl", receipts)
    normalize_recon_raw(run, config)


def _report_result(job: dict[str, object], decision: str, *, evidence: dict[str, object] | None = None, reason: str | None = None) -> str:
    import json

    priors = [] if evidence is None else [{
        "paper": evidence["title"], "exact_overlap": "same failure regime",
        "residual_difference": "different control mechanism", "evidence_url_or_id": evidence["evidence_id"],
    }]
    if reason is None:
        reason = {
            "COVERED": f"The normalized prior {evidence['title']} directly covers the residual." if evidence else "missing evidence",
            "NEAR_PRIOR_WITH_RESIDUAL": f"The normalized prior {evidence['title']} leaves the stated residual." if evidence else "missing evidence",
            "NO_DIRECT_COVERAGE_FOUND": "No direct coverage found under this query pack. This result is bounded to the supplied queries and execution receipts.",
        }[decision]
    return json.dumps({
        "schema_version": "idea_factory.recon_report_result.v1", "job_id": job["job_id"],
        "opportunity_id": job["opportunity_id"], "query_pack_sha256": job["query_pack_sha256"],
        "normalized_evidence_manifest_sha256": job["normalized_evidence_manifest_sha256"],
        "protocol_hash": job["protocol_hash"], "cache_key": job["cache_key"], "searched_query_ids": job["searched_query_ids"],
        "searched_at": "2026-08-03T11:00:00+08:00", "nearest_priors": priors,
        "decision": decision, "decision_reason": reason,
    })


def test_report_jobs_replay_exact_and_contain_no_raw_payloads(tmp_path: Path) -> None:
    import json
    import pytest
    from idea_factory.artifacts import read_jsonl
    from idea_factory.recon import emit_recon_report_jobs, validate_report_jobs

    run, config = _build_dedup_run(tmp_path); _emit_normalized_chain(run, config, with_evidence=True)
    path = emit_recon_report_jobs(run, config)
    jobs = validate_report_jobs(run, config)
    assert len(jobs) == 1
    serialized = json.dumps(jobs)
    assert "abstract_inverted_index" not in serialized and "raw_result_json" not in serialized and '"papers"' not in serialized
    rows = read_jsonl(path); rows[0]["protocol_hash"] = "0" * 64
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="report job replay"):
        validate_report_jobs(run, config)


def test_report_alias_reference_survives_full_chain_and_replays_exactly(tmp_path: Path) -> None:
    import hashlib
    import json
    from idea_factory.artifacts import read_jsonl, write_jsonl
    from idea_factory.recon import emit_recon_report_jobs, ingest_recon_reports, normalize_recon_raw, validate_recon_bundle, validate_report_jobs

    run, config = _build_dedup_run(tmp_path); jobs, receipts = _emit_empty_execution_chain(run, config)
    query_id = jobs[0]["query_id"]
    arxiv_job = next(row for row in jobs if row["source"] == "ARXIV" and row["query_id"] == query_id)
    openalex_job = next(row for row in jobs if row["source"] == "OPENALEX" and row["query_id"] == query_id)
    arxiv_paper = _arxiv_paper("http://arxiv.org/abs/2501.00001v1", "Shared prior", "https://arxiv.org/pdf/2501.00001")
    arxiv_paper["doi"] = "10.1/shared"
    payloads = {
        arxiv_job["job_id"]: json.dumps({"status": "success", "results_count": 1, "papers": [arxiv_paper]}).encode(),
        openalex_job["job_id"]: json.dumps({
            "meta": {"count": 1, "per_page": 10},
            "results": [{
                "id": "https://openalex.org/W1", "display_name": "Shared prior", "publication_year": 2025,
                "doi": "https://doi.org/10.1/shared", "authorships": [], "abstract_inverted_index": None,
                "primary_location": {"landing_page_url": "https://example.test/shared"},
            }],
        }).encode(),
    }
    for job in (arxiv_job, openalex_job):
        body = payloads[job["job_id"]]; (run / job["raw_output_path"]).write_bytes(body)
        receipt = next(row for row in receipts if row["job_id"] == job["job_id"])
        receipt["status"] = "SUCCESS"; receipt["raw_file_sha256"] = hashlib.sha256(body).hexdigest()
    write_jsonl(run / "recon" / "execution_receipts.jsonl", receipts)
    normalize_recon_raw(run, config)
    jobs_path = emit_recon_report_jobs(run, config); report_job = read_jsonl(jobs_path)[0]
    evidence = report_job["evidence"][0]
    assert evidence["source_aliases"] == sorted(evidence["source_aliases"])
    assert evidence["source_aliases"] == ["ARXIV:http://arxiv.org/abs/2501.00001v1", "OPENALEX:https://openalex.org/W1"]
    raw_report = json.loads(_report_result(report_job, "COVERED", evidence=evidence))
    raw_report["nearest_priors"][0]["evidence_url_or_id"] = evidence["source_aliases"][1]
    ingest_recon_reports(jobs_path, [json.dumps(raw_report)], config)
    assert validate_recon_bundle(run, config)["killed"]
    rows = read_jsonl(jobs_path); rows[0]["evidence"][0]["source_aliases"].reverse()
    write_jsonl(jobs_path, rows)
    with pytest.raises(ValueError, match="report job replay"):
        validate_report_jobs(run, config)


def test_full_recon_bundle_accepts_all_three_bounded_decisions(tmp_path: Path) -> None:
    from idea_factory.artifacts import read_jsonl
    from idea_factory.recon import emit_recon_report_jobs, ingest_recon_reports, validate_recon_bundle

    for index, decision in enumerate(("COVERED", "NEAR_PRIOR_WITH_RESIDUAL", "NO_DIRECT_COVERAGE_FOUND")):
        case = tmp_path / str(index); case.mkdir()
        run, config = _build_dedup_run(case); _emit_normalized_chain(run, config, with_evidence=decision != "NO_DIRECT_COVERAGE_FOUND")
        jobs_path = emit_recon_report_jobs(run, config); job = read_jsonl(jobs_path)[0]
        evidence = job["evidence"][0] if job["evidence"] else None
        ingest_recon_reports(jobs_path, [_report_result(job, decision, evidence=evidence)], config)
        bundle = validate_recon_bundle(run, config)
        assert bundle["reports"][0]["decision"] == decision
        assert len(bundle["killed"] if decision == "COVERED" else bundle["ready"]) == 1


def test_report_rejects_missing_lane_invented_evidence_and_global_novelty(tmp_path: Path) -> None:
    import json
    from idea_factory.artifacts import read_jsonl, write_jsonl
    from idea_factory.recon import emit_recon_report_jobs, ingest_recon_reports

    run, config = _build_dedup_run(tmp_path); _emit_normalized_chain(run, config, with_evidence=True)
    jobs_path = emit_recon_report_jobs(run, config); job = read_jsonl(jobs_path)[0]
    raw = json.loads(_report_result(job, "NO_DIRECT_COVERAGE_FOUND", reason="This is novel and no direct coverage found under this query pack"))
    ingest_recon_reports(jobs_path, [json.dumps(raw)], config)
    assert read_jsonl(run / "recon" / "rejected.jsonl")


@pytest.mark.parametrize(
    "global_claim",
    [
        "No one has ever done this.",
        "This is the first solution.",
        "It is globally the highest priority direction.",
        "Across all literature, nobody has attempted it.",
    ],
)
def test_no_direct_coverage_uses_closed_literal_reason_without_global_claims(tmp_path: Path, global_claim: str) -> None:
    import json
    from idea_factory.artifacts import read_jsonl
    from idea_factory.recon import emit_recon_report_jobs, ingest_recon_reports

    run, config = _build_dedup_run(tmp_path); _emit_normalized_chain(run, config, with_evidence=False)
    jobs_path = emit_recon_report_jobs(run, config); job = read_jsonl(jobs_path)[0]
    bounded = "no direct coverage found under this query pack; evidence is bounded to these queries"
    raw = _report_result(job, "NO_DIRECT_COVERAGE_FOUND", reason=f"{bounded}. {global_claim}")
    ingest_recon_reports(jobs_path, [raw], config)
    error = read_jsonl(run / "recon" / "rejected.jsonl")[0]["error"]
    assert "controlled literal" in error


def test_report_rejects_invented_evidence_reference(tmp_path: Path) -> None:
    import json
    from idea_factory.artifacts import read_jsonl
    from idea_factory.recon import emit_recon_report_jobs, ingest_recon_reports

    run, config = _build_dedup_run(tmp_path); _emit_normalized_chain(run, config, with_evidence=True)
    jobs_path = emit_recon_report_jobs(run, config); job = read_jsonl(jobs_path)[0]
    raw = json.loads(_report_result(job, "COVERED", evidence=job["evidence"][0]))
    raw["nearest_priors"][0]["evidence_url_or_id"] = "invented-evidence-id"
    ingest_recon_reports(jobs_path, [json.dumps(raw)], config)
    assert "evidence-bound" in read_jsonl(run / "recon" / "rejected.jsonl")[0]["error"]


def test_rejected_report_outcome_never_persists_raw_secret(tmp_path: Path) -> None:
    import json
    from idea_factory.artifacts import read_jsonl
    from idea_factory.recon import emit_recon_report_jobs, ingest_recon_reports, validate_recon_bundle

    sentinel = "SECRET_SENTINEL_DO_NOT_PERSIST"
    run, config = _build_dedup_run(tmp_path); _emit_normalized_chain(run, config, with_evidence=False)
    jobs_path = emit_recon_report_jobs(run, config); job = read_jsonl(jobs_path)[0]
    raw = json.loads(_report_result(job, "NO_DIRECT_COVERAGE_FOUND")); raw["secret"] = sentinel
    ingest_recon_reports(jobs_path, [json.dumps(raw)], config)
    serialized = json.dumps({
        "outcomes": read_jsonl(run / "recon" / "outcomes.jsonl"),
        "rejected": read_jsonl(run / "recon" / "rejected.jsonl"),
    })
    assert sentinel not in serialized
    outcome = read_jsonl(run / "recon" / "outcomes.jsonl")[0]
    assert outcome["status"] == "REJECTED" and "raw_result_json" not in outcome
    assert validate_recon_bundle(run, config)["rejected"]


def test_nearest_prior_paper_must_equal_canonical_evidence_title(tmp_path: Path) -> None:
    import json
    from idea_factory.artifacts import read_jsonl
    from idea_factory.recon import emit_recon_report_jobs, ingest_recon_reports

    run, config = _build_dedup_run(tmp_path); _emit_normalized_chain(run, config, with_evidence=True)
    jobs_path = emit_recon_report_jobs(run, config); job = read_jsonl(jobs_path)[0]
    raw = json.loads(_report_result(job, "COVERED", evidence=job["evidence"][0]))
    raw["nearest_priors"][0]["paper"] = "Forged display title"
    raw["decision_reason"] = "Forged display title directly covers the residual."
    ingest_recon_reports(jobs_path, [json.dumps(raw)], config)
    assert "canonical title" in read_jsonl(run / "recon" / "rejected.jsonl")[0]["error"]


def test_covered_reason_must_name_its_bound_prior(tmp_path: Path) -> None:
    import json
    from idea_factory.artifacts import read_jsonl
    from idea_factory.recon import emit_recon_report_jobs, ingest_recon_reports

    run, config = _build_dedup_run(tmp_path); _emit_normalized_chain(run, config, with_evidence=True)
    jobs_path = emit_recon_report_jobs(run, config); job = read_jsonl(jobs_path)[0]
    raw = _report_result(job, "COVERED", evidence=job["evidence"][0], reason="Trust this conclusion.")
    ingest_recon_reports(jobs_path, [raw], config)
    assert "decision reason" in read_jsonl(run / "recon" / "rejected.jsonl")[0]["error"]


def test_report_rejects_lane_with_no_success_or_empty_receipt(tmp_path: Path) -> None:
    from idea_factory.artifacts import read_jsonl, write_jsonl
    from idea_factory.recon import emit_recon_report_jobs, ingest_recon_reports, normalize_recon_raw

    run, config = _build_dedup_run(tmp_path); _jobs, receipts = _emit_empty_execution_chain(run, config)
    query_rows = read_jsonl(run / "recon" / "query_pack.jsonl")
    evaluation_ids = {row["query_id"] for row in query_rows if row["lane"] == "EVALUATION"}
    for receipt in receipts:
        if receipt["query_id"] in evaluation_ids:
            receipt["status"] = "ERROR"; receipt["http_status"] = 500; receipt["raw_file_sha256"] = None
            receipt["exit_code"] = 1
            receipt["error_reason"] = "tool error"
    write_jsonl(run / "recon" / "execution_receipts.jsonl", receipts)
    normalize_recon_raw(run, config)
    jobs_path = emit_recon_report_jobs(run, config); job = read_jsonl(jobs_path)[0]
    ingest_recon_reports(jobs_path, [_report_result(job, "NO_DIRECT_COVERAGE_FOUND")], config)
    rejected = read_jsonl(run / "recon" / "rejected.jsonl")
    outcomes = read_jsonl(run / "recon" / "outcomes.jsonl")
    assert rejected[0]["error_code"] == "LANE_COVERAGE_INCOMPLETE"
    assert "each recon lane" in rejected[0]["error"]
    assert outcomes[0]["status"] == "REJECTED"
    assert outcomes[0]["error_code"] == "LANE_COVERAGE_INCOMPLETE"
    assert read_jsonl(run / "recon" / "ready_for_routes.jsonl") == []


def test_report_ingestion_failure_cleans_stale_publications(tmp_path: Path) -> None:
    import pytest
    from idea_factory.artifacts import read_jsonl
    from idea_factory.recon import emit_recon_report_jobs, ingest_recon_reports

    run, config = _build_dedup_run(tmp_path); _emit_normalized_chain(run, config, with_evidence=False)
    jobs_path = emit_recon_report_jobs(run, config); job = read_jsonl(jobs_path)[0]
    good = _report_result(job, "NO_DIRECT_COVERAGE_FOUND")
    ingest_recon_reports(jobs_path, [good], config)
    with pytest.raises(ValueError, match="duplicate"):
        ingest_recon_reports(jobs_path, [good, good], config)
    assert not (run / "recon" / "reports.jsonl").exists()
    assert not (run / "recon" / "ready_for_routes.jsonl").exists()


def test_recon_report_ingestion_and_validation_are_idempotent(tmp_path: Path) -> None:
    from idea_factory.artifacts import read_jsonl
    from idea_factory.recon import emit_recon_report_jobs, ingest_recon_reports, validate_recon_bundle

    run, config = _build_dedup_run(tmp_path); _emit_normalized_chain(run, config, with_evidence=False)
    jobs_path = emit_recon_report_jobs(run, config); job = read_jsonl(jobs_path)[0]
    raw = _report_result(job, "NO_DIRECT_COVERAGE_FOUND")
    ingest_recon_reports(jobs_path, [raw], config)
    names = ("reports.jsonl", "rejected.jsonl", "outcomes.jsonl", "ready_for_routes.jsonl", "killed.jsonl")
    first = {name: (run / "recon" / name).read_bytes() for name in names}
    ingest_recon_reports(jobs_path, [raw], config); validate_recon_bundle(run, config)
    assert first == {name: (run / "recon" / name).read_bytes() for name in names}


def test_query_regeneration_rejects_junction_without_touching_outside(tmp_path: Path) -> None:
    import os
    import pytest
    from idea_factory.recon import emit_recon_query_pack

    run, config = _build_dedup_run(tmp_path)
    recon = run / "recon"; recon.mkdir()
    outside = tmp_path / "outside"; outside.mkdir(); sentinel = outside / "sentinel.txt"; sentinel.write_text("keep", encoding="utf-8")
    try:
        os.symlink(outside, recon / "raw", target_is_directory=True)
    except OSError:
        import subprocess
        completed = subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(recon / "raw"), str(outside)], capture_output=True)
        if completed.returncode:
            pytest.skip("directory links unavailable")
    with pytest.raises(ValueError, match="escapes|anchored"):
        emit_recon_query_pack(run, config)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_query_regeneration_rejects_nested_junction_before_any_cleanup(tmp_path: Path) -> None:
    import os
    import subprocess
    import pytest
    from idea_factory.recon import emit_recon_query_pack

    run, config = _build_dedup_run(tmp_path)
    recon = run / "recon"; normalized = recon / "normalized"; normalized.mkdir(parents=True)
    outside = tmp_path / "outside-nested"; outside.mkdir(); sentinel = outside / "sentinel.txt"; sentinel.write_text("keep", encoding="utf-8")
    link = normalized / "escape"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except OSError:
        completed = subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(link), str(outside)], capture_output=True)
        if completed.returncode:
            pytest.skip("directory links unavailable")
    stale = recon / "reports.jsonl"; stale.write_text('{"stale":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="escapes|anchored"):
        emit_recon_query_pack(run, config)
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert stale.exists()
