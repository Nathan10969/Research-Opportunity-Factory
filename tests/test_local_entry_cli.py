from pathlib import Path

import pytest


def test_only_emit_opportunity_parser_accepts_local_option():
    from idea_factory.cli import build_parser
    args = build_parser().parse_args(["emit-opportunity-jobs", "--run", "r", "--reviewed-local-entries", "local.json"])
    assert args.reviewed_local_entries == Path("local.json")


@pytest.mark.parametrize("command", ["emit-corpus-jobs", "build-landscape", "quality-gate", "ingest-opportunities"])
def test_other_commands_reject_local_option(command):
    from idea_factory.cli import build_parser
    with pytest.raises(SystemExit):
        build_parser().parse_args([command, "--run", "r", "--reviewed-local-entries", "local.json"])


def test_cli_forwards_reviewed_local_entries(monkeypatch, tmp_path):
    from idea_factory import pipeline
    from idea_factory.cli import main
    captured = {}
    monkeypatch.setattr(pipeline, "execute", lambda command, run, config, **kwargs: captured.update(kwargs) or "NO_OP")
    assert main(["emit-opportunity-jobs", "--run", str(tmp_path), "--reviewed-local-entries", "local.json"]) == 0
    assert captured["reviewed_local_entries"] == Path("local.json")


def test_direct_pipeline_rejects_local_option_on_wrong_command(monkeypatch, tmp_path):
    from idea_factory import pipeline
    monkeypatch.setattr(pipeline, "_anchored_run", lambda path: tmp_path)
    with pytest.raises(pipeline.PipelineError, match="only valid"):
        pipeline._execute_unlocked("quality-gate", tmp_path, None, reviewed_local_entries=tmp_path / "local.json")


def _prepared_pipeline_run(tmp_path, monkeypatch):
    from idea_factory import pipeline
    from test_pipeline import FixtureBackend, _config, _jsonl
    config = _config(tmp_path); run = tmp_path / "run"; result = tmp_path / "result.jsonl"
    _jsonl(result, [{"fixture": "v1"}])
    monkeypatch.setattr(pipeline, "BACKEND", FixtureBackend())
    pipeline.init_run(run, config, mode="offline-fixture")
    for command, kwargs in (("emit-corpus-jobs", {}), ("ingest-corpus-labels", {"results": result}), ("emit-card-jobs", {}), ("ingest-cards", {"results": result}), ("build-landscape", {})):
        pipeline.execute(command, run, config, **kwargs)
    return pipeline, run, config


def test_execute_local_option_noop_and_changed_input_refusal(monkeypatch, tmp_path):
    pipeline, run, config = _prepared_pipeline_run(tmp_path, monkeypatch)
    local = tmp_path / "local.json"; local.write_text("{}", encoding="utf-8")
    assert pipeline.execute("emit-opportunity-jobs", run, config, reviewed_local_entries=local) == "COMPLETED"
    state_before = (run / "state.json").read_bytes()
    assert pipeline.execute("emit-opportunity-jobs", run, config, reviewed_local_entries=local) == "NO_OP"
    assert (run / "state.json").read_bytes() == state_before
    local.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(pipeline.PipelineError, match="new run|changed"):
        pipeline.execute("emit-opportunity-jobs", run, config, reviewed_local_entries=local)
    assert (run / "state.json").read_bytes() == state_before


def test_execute_without_local_option_omits_digest_key_and_none_is_identical(monkeypatch, tmp_path):
    pipeline, run, config = _prepared_pipeline_run(tmp_path, monkeypatch)
    observed = []
    original = pipeline._command_digest
    def spy(command, config_path, paths, options):
        observed.append(dict(paths))
        return original(command, config_path, paths, options)
    monkeypatch.setattr(pipeline, "_command_digest", spy)
    assert pipeline.execute("emit-opportunity-jobs", run, config) == "COMPLETED"
    assert "reviewed_local_entries" not in observed[-1]
    state = (run / "state.json").read_bytes()
    assert pipeline.execute("emit-opportunity-jobs", run, config, reviewed_local_entries=None) == "NO_OP"
    assert (run / "state.json").read_bytes() == state
