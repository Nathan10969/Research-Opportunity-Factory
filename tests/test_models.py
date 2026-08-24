from collections import UserDict
from copy import deepcopy
from types import MappingProxyType

import pytest
from pydantic import ValidationError, model_validator

from idea_factory.models import (
    ArtifactEvidence,
    CompletionKind,
    CompletionManifest,
    DecisiveTest,
    EvidencePointer,
    FailureClaim,
    FrozenStrictModel,
    HumanScores,
    IdeaPack,
    IdeaStatus,
    IdeaYieldEvidence,
    NearestPrior,
    Opportunity,
    OpportunityOperator,
    ReconQuery,
    RUN_ORDER,
    RunStage,
    RunState,
    ZERO_SURVIVOR_STAGES,
    ZeroSurvivorEvidence,
)


class DerivedFrozenRecord(FrozenStrictModel):
    x: int
    doubled: int = 0

    @model_validator(mode="after")
    def derive_doubled(self) -> "DerivedFrozenRecord":
        object.__setattr__(self, "doubled", self.x * 2)
        return self


def opportunity() -> Opportunity:
    return Opportunity(
        schema_version="idea_factory.opportunity.v1",
        opportunity_id="opp-1",
        operator=OpportunityOperator.FAILURE_TRANSFER,
        assumption_x="fixed cache budget is sufficient",
        observation_y="reuse degrades after eviction",
        condition_z="long-running agent sessions",
        failure_f="retrieval misses useful context",
        missing_capability_w="adaptive retention",
        alternative_explanation_a="the benchmark distribution shifted",
        decisive_experiment="compare fixed and adaptive retention",
        supporting_card_ids=["card-1"],
        nearest_internal_neighbors=[],
        scope_compatibility="same workload regime",
    )


def nearest_prior() -> NearestPrior:
    return NearestPrior(
        paper="Prior adaptive cache paper",
        exact_overlap="adaptive retention",
        residual_difference="does not cover agent memory",
        evidence_url_or_id="doi:example",
    )


def ready_pack(**overrides: object) -> IdeaPack:
    data = {
        "idea_id": "idea-1",
        "status": IdeaStatus.READY_FOR_CHEAP_TEST,
        "opportunity": "opp-1",
        "core_hypothesis": "adaptive retention improves recall",
        "why_now": "new traces expose the failure",
        "proposed_mechanism": "retention controller",
        "source_of_gain": "reduced harmful eviction",
        "cheapest_decisive_test": DecisiveTest(
            setup="replay 100 agent traces",
            discriminates_against="static retention",
            expected_runtime_or_cost="one GPU hour",
        ),
        "strongest_baseline": "best static retention policy",
        "kill_condition": "no recall improvement at matched cost",
        "main_uncertainty": "trace diversity",
        "expected_reviewer_2_objection": "this is an old cache policy",
        "response_to_objection": "the setting differs",
        "evidence_that_would_make_reviewer_correct": "same result in matched setting",
        "nearest_priors": [nearest_prior()],
        "human_scores": HumanScores(
            specific_novelty=4,
            importance=3,
            paper_potential=3,
            feasibility=5,
            excitement=3,
            evidence_clarity=4,
        ),
        "human_reason_codes": ["human-reviewed"],
    }
    data.update(overrides)
    return IdeaPack(**data)


def test_unknown_evidence_status_is_rejected() -> None:
    with pytest.raises(ValidationError):
        FailureClaim(text="claim", status="UNSUPPORTED")


def test_ready_pack_missing_readiness_fields_is_rejected() -> None:
    with pytest.raises(ValidationError, match="source_of_gain"):
        ready_pack(
            source_of_gain=" ",
            cheapest_decisive_test=None,
            strongest_baseline=" ",
            kill_condition=" ",
            nearest_priors=[],
            human_scores=None,
            human_reason_codes=[],
        )


def test_valid_ready_pack_serializes_enum_values_as_json_strings() -> None:
    payload = ready_pack().model_dump(mode="json")
    opportunity_payload = opportunity().model_dump(mode="json")

    assert payload["status"] == "READY_FOR_CHEAP_TEST"
    assert payload["opportunity"] == "opp-1"
    assert isinstance(payload["nearest_priors"], list)
    assert isinstance(payload["human_reason_codes"], list)
    assert opportunity_payload["operator"] == "FAILURE_TRANSFER"


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EvidencePointer(
            source="NOTE", locator="p. 1", supports="claim", unexpected="nope"
        )


@pytest.mark.parametrize("value", [-1, 6])
def test_human_scores_reject_values_outside_zero_to_five(value: int) -> None:
    with pytest.raises(ValidationError):
        HumanScores(
            specific_novelty=value,
            importance=3,
            paper_potential=3,
            feasibility=3,
            excitement=3,
            evidence_clarity=3,
        )


@pytest.mark.parametrize("max_results", [0, 11])
def test_recon_query_rejects_out_of_range_max_results(max_results: int) -> None:
    with pytest.raises(ValidationError):
        ReconQuery(
            query_id="query-1",
            opportunity_id="opp-1",
            lane="CONCEPT",
            variant="CURRENT_TERMS",
            query="adaptive retention agent memory",
            max_results=max_results,
        )


def test_collection_defaults_are_immutable_and_empty_across_instances() -> None:
    first = IdeaPack(
        idea_id="idea-1",
        status=IdeaStatus.HOLD,
        opportunity="opp-1",
        core_hypothesis="h1",
        why_now="now",
        proposed_mechanism="mechanism",
        source_of_gain="",
        strongest_baseline="baseline",
        kill_condition="condition",
        main_uncertainty="uncertainty",
        expected_reviewer_2_objection="objection",
        response_to_objection="response",
        evidence_that_would_make_reviewer_correct="evidence",
    )
    second = IdeaPack(
        idea_id="idea-2",
        status=IdeaStatus.HOLD,
        opportunity="opp-2",
        core_hypothesis="h2",
        why_now="now",
        proposed_mechanism="mechanism",
        source_of_gain="",
        strongest_baseline="baseline",
        kill_condition="condition",
        main_uncertainty="uncertainty",
        expected_reviewer_2_objection="objection",
        response_to_objection="response",
        evidence_that_would_make_reviewer_correct="evidence",
    )

    assert first.supporting_observations == second.supporting_observations == ()
    assert first.inference_flags == second.inference_flags == ()
    assert first.nearest_priors == second.nearest_priors == ()
    assert first.human_reason_codes == second.human_reason_codes == ()


def test_invalid_assignment_to_ready_status_preserves_hold_status() -> None:
    pack = ready_pack(
        status=IdeaStatus.HOLD,
        source_of_gain="",
        cheapest_decisive_test=None,
        strongest_baseline="",
        kill_condition="",
        nearest_priors=[],
        human_scores=None,
        human_reason_codes=[],
    )

    with pytest.raises(ValidationError, match="frozen"):
        pack.status = IdeaStatus.READY_FOR_CHEAP_TEST

    assert pack.status == IdeaStatus.HOLD


def test_invalid_ready_field_assignment_preserves_original_value() -> None:
    pack = ready_pack()

    with pytest.raises(ValidationError, match="frozen"):
        pack.source_of_gain = " "

    assert pack.source_of_gain == "reduced harmful eviction"


def test_coercible_byte_ready_status_cannot_bypass_readiness_validation() -> None:
    with pytest.raises(ValidationError, match="source_of_gain"):
        ready_pack(
            status=b"READY_FOR_CHEAP_TEST",
            source_of_gain="",
            cheapest_decisive_test=None,
            strongest_baseline="",
            kill_condition="",
            nearest_priors=[],
            human_scores=None,
            human_reason_codes=[],
        )


def test_mapping_input_cannot_bypass_readiness_validation() -> None:
    data = ready_pack().model_dump()
    data.update(
        source_of_gain="",
        cheapest_decisive_test=None,
        strongest_baseline="",
        kill_condition="",
        nearest_priors=[],
        human_scores=None,
        human_reason_codes=[],
    )

    with pytest.raises(ValidationError, match="source_of_gain"):
        IdeaPack.model_validate(UserDict(data))


def test_empty_iterators_cannot_create_invalid_ready_pack() -> None:
    with pytest.raises(ValidationError, match="nearest_priors.*human_reason_codes"):
        ready_pack(nearest_priors=iter(()), human_reason_codes=iter(()))


def test_ready_pack_collections_are_immutable() -> None:
    pack = ready_pack()

    assert isinstance(pack.supporting_observations, tuple)
    assert isinstance(pack.inference_flags, tuple)
    assert isinstance(pack.nearest_priors, tuple)
    assert isinstance(pack.human_reason_codes, tuple)

    with pytest.raises(AttributeError):
        pack.nearest_priors.clear()
    with pytest.raises(AttributeError):
        pack.nearest_priors.pop()
    with pytest.raises(TypeError):
        pack.nearest_priors[0] = nearest_prior()
    with pytest.raises(AttributeError):
        pack.human_reason_codes.append(7)


def test_ready_pack_nested_evidence_is_frozen() -> None:
    pack = ready_pack()

    with pytest.raises(ValidationError, match="frozen"):
        pack.nearest_priors[0].paper = "changed"
    with pytest.raises(ValidationError, match="frozen"):
        pack.cheapest_decisive_test.setup = "changed"
    with pytest.raises(ValidationError, match="frozen"):
        pack.human_scores.importance = 0


def test_model_copy_update_cannot_create_invalid_ready_pack() -> None:
    pack = ready_pack()

    with pytest.raises(ValidationError, match="source_of_gain"):
        pack.model_copy(update={"source_of_gain": " "})


@pytest.mark.parametrize("field_name", ["nearest_priors", "human_reason_codes"])
def test_model_copy_update_materializes_empty_iterator_before_readiness_gate(
    field_name: str,
) -> None:
    pack = ready_pack()

    with pytest.raises(ValidationError, match=field_name):
        pack.model_copy(update={field_name: iter(())})


@pytest.mark.parametrize(
    "field_name",
    ["paper", "exact_overlap", "residual_difference", "evidence_url_or_id"],
)
def test_nearest_prior_rejects_blank_evidence_fields(field_name: str) -> None:
    data = nearest_prior().model_dump()
    data[field_name] = " "

    with pytest.raises(ValidationError):
        NearestPrior(**data)


@pytest.mark.parametrize(
    "field_name", ["setup", "discriminates_against", "expected_runtime_or_cost"]
)
def test_decisive_test_rejects_blank_evidence_fields(field_name: str) -> None:
    data = {
        "setup": "replay traces",
        "discriminates_against": "static policy",
        "expected_runtime_or_cost": "one hour",
    }
    data[field_name] = " "

    with pytest.raises(ValidationError):
        DecisiveTest(**data)


def test_ready_pack_rejects_blank_human_reason_code() -> None:
    with pytest.raises(ValidationError):
        ready_pack(human_reason_codes=[" "])


def test_idea_pack_accepts_opportunity_identifier_string() -> None:
    pack = ready_pack(opportunity="opp-1")

    assert pack.opportunity == "opp-1"


def test_idea_pack_rejects_embedded_opportunity_object() -> None:
    with pytest.raises(ValidationError):
        ready_pack(opportunity=opportunity())


@pytest.mark.parametrize(
    "reason_codes",
    [{"reviewed": True}, UserDict({"reviewed": True})],
)
def test_ready_pack_rejects_mapping_human_reason_codes(reason_codes: object) -> None:
    with pytest.raises(ValidationError):
        ready_pack(human_reason_codes=reason_codes)


def test_ready_pack_rejects_mapping_keyed_by_nearest_prior() -> None:
    with pytest.raises(ValidationError):
        ready_pack(nearest_priors={nearest_prior(): True})


def test_exploding_iterator_is_reported_as_validation_error() -> None:
    def exploding_reason_codes():
        yield "reviewed"
        raise RuntimeError("iteration exploded")

    with pytest.raises(ValidationError):
        ready_pack(human_reason_codes=exploding_reason_codes())


def test_noop_model_copy_preserves_unset_fields_and_shallow_nested_identity() -> None:
    pack = ready_pack()

    copied = pack.model_copy()

    assert copied.model_dump(exclude_unset=True) == pack.model_dump(
        exclude_unset=True
    )
    assert copied.model_fields_set == pack.model_fields_set
    assert copied.nearest_priors[0] is pack.nearest_priors[0]
    assert copied.cheapest_decisive_test is pack.cheapest_decisive_test
    assert copied.human_scores is pack.human_scores


def test_validated_shallow_model_copy_preserves_fields_set_and_nested_identity() -> None:
    pack = ready_pack()

    copied = pack.model_copy(update={"supporting_observations": ("observation",)})

    assert copied.model_fields_set == pack.model_fields_set | {
        "supporting_observations"
    }
    assert copied.nearest_priors[0] is pack.nearest_priors[0]
    assert copied.cheapest_decisive_test is pack.cheapest_decisive_test
    assert copied.human_scores is pack.human_scores


def test_nearest_prior_model_copy_rejects_blank_update() -> None:
    with pytest.raises(ValidationError):
        nearest_prior().model_copy(update={"paper": " "})


def test_decisive_test_model_copy_rejects_blank_update() -> None:
    decisive_test = ready_pack().cheapest_decisive_test

    with pytest.raises(ValidationError):
        decisive_test.model_copy(update={"setup": " "})


def test_human_scores_model_copy_rejects_out_of_range_update() -> None:
    scores = ready_pack().human_scores

    with pytest.raises(ValidationError):
        scores.model_copy(update={"specific_novelty": 99})


def test_idea_pack_rejects_invalid_constructed_nearest_prior() -> None:
    data = nearest_prior().model_dump()
    data["paper"] = " "
    invalid_prior = NearestPrior.model_construct(**data)

    with pytest.raises(ValidationError):
        ready_pack(nearest_priors=(invalid_prior,))


def test_idea_pack_rejects_invalid_constructed_decisive_test() -> None:
    invalid_test = DecisiveTest.model_construct(
        setup=" ",
        discriminates_against="static policy",
        expected_runtime_or_cost="one hour",
    )

    with pytest.raises(ValidationError):
        ready_pack(cheapest_decisive_test=invalid_test)


def test_idea_pack_rejects_invalid_constructed_human_scores() -> None:
    invalid_scores = HumanScores.model_construct(
        specific_novelty=99,
        importance=3,
        paper_potential=3,
        feasibility=3,
        excitement=3,
        evidence_clarity=3,
    )

    with pytest.raises(ValidationError):
        ready_pack(human_scores=invalid_scores)


def test_validated_model_copy_preserves_after_validator_derived_fields() -> None:
    record = DerivedFrozenRecord(x=1)

    copied = record.model_copy(update={"x": 2})

    assert record.doubled == 2
    assert copied.x == 2
    assert copied.doubled == 4


def test_run_stage_order_and_zero_survivor_stage_contract() -> None:
    assert RUN_ORDER == (
        RunStage.CREATED,
        RunStage.CORPUS_ROUTED,
        RunStage.CARDS_READY,
        RunStage.LANDSCAPE_READY,
        RunStage.OPPORTUNITIES_READY,
        RunStage.QUALITY_GATED,
        RunStage.RECON_READY,
        RunStage.ROUTES_READY,
        RunStage.REVIEWED,
        RunStage.HUMAN_SCORED,
        RunStage.COMPLETE,
    )
    assert ZERO_SURVIVOR_STAGES == (
        RunStage.QUALITY_GATED,
        RunStage.RECON_READY,
        RunStage.REVIEWED,
    )


def test_run_state_advances_only_one_normal_stage_at_a_time() -> None:
    state = RunState(run_id="run-1")

    for stage in RUN_ORDER[1:-1]:
        state.advance(stage)
        assert state.stage == stage

    assert state.stage == RunStage.HUMAN_SCORED
    assert state.completion_kind is None


@pytest.mark.parametrize(
    "target",
    [RunStage.CREATED, RunStage.CARDS_READY, RunStage.COMPLETE],
)
def test_run_state_rejects_noop_skip_and_unproven_completion(target: RunStage) -> None:
    state = RunState(run_id="run-1")

    with pytest.raises(ValueError):
        state.advance(target)


def test_completed_run_state_rejects_advance_with_clear_error() -> None:
    state = RunState(run_id="run-1", stage=RunStage.QUALITY_GATED)
    state.advance(RunStage.COMPLETE, zero_survivor=True)

    with pytest.raises(ValueError, match="already COMPLETE"):
        state.advance(RunStage.COMPLETE)


@pytest.mark.parametrize("stage", ZERO_SURVIVOR_STAGES)
def test_zero_survivor_completion_is_allowed_only_from_explicit_stages(stage: RunStage) -> None:
    state = RunState(run_id="run-1", stage=stage)

    state.advance(RunStage.COMPLETE, zero_survivor=True)

    assert state.stage == RunStage.COMPLETE
    assert state.completion_kind == CompletionKind.AUDITABLE_ZERO_SURVIVOR


@pytest.mark.parametrize(
    "stage", [RunStage.CREATED, RunStage.CARDS_READY, RunStage.ROUTES_READY, RunStage.HUMAN_SCORED]
)
def test_zero_survivor_completion_rejects_other_stages(stage: RunStage) -> None:
    state = RunState(run_id="run-1", stage=stage)

    with pytest.raises(ValueError):
        state.advance(RunStage.COMPLETE, zero_survivor=True)


def test_normal_completion_requires_human_confirmed_ready_pack() -> None:
    state = RunState(
        run_id="run-1",
        stage=RunStage.HUMAN_SCORED,
        input_hashes={"corpus": "abc123"},
    )

    with pytest.raises(ValueError, match="human-confirmed READY"):
        state.finalize_idea_yield([])

    state.finalize_idea_yield([ready_pack()])

    assert state.stage == RunStage.COMPLETE
    assert state.completion_kind == CompletionKind.IDEA_YIELD
    assert isinstance(state.completion_evidence, IdeaYieldEvidence)
    assert state.completion_evidence.human_confirmed_ready_packs == (ready_pack(),)
    assert state.completion_evidence.input_hash_snapshot == (("corpus", "abc123"),)
    assert state.completion_evidence.ready_pack_artifact.record_ids == ("idea-1",)
    assert state.completion_evidence.ready_pack_artifact.record_count == 1
    assert state.completion_evidence.human_attestation_artifact.record_ids == (
        "idea-1",
    )
    assert len(state.completion_evidence.manifest_sha256) == 64


def test_unresolved_errors_block_both_completion_kinds() -> None:
    normal = RunState(
        run_id="run-1", stage=RunStage.HUMAN_SCORED, unresolved_errors=["API timeout"]
    )
    zero = RunState(
        run_id="run-2", stage=RunStage.QUALITY_GATED, unresolved_errors=["API timeout"]
    )

    with pytest.raises(ValueError, match="unresolved errors"):
        normal.finalize_idea_yield([ready_pack()])
    with pytest.raises(ValueError, match="unresolved errors"):
        zero.advance(RunStage.COMPLETE, zero_survivor=True)


def test_completion_construction_requires_matching_auditable_evidence() -> None:
    with pytest.raises(ValidationError, match="completion_evidence"):
        RunState(
            run_id="run-1",
            stage=RunStage.COMPLETE,
            completion_kind=CompletionKind.IDEA_YIELD,
        )
    with pytest.raises(ValidationError, match="completion_evidence"):
        RunState.model_validate(
            {
                "run_id": "run-1",
                "stage": "COMPLETE",
                "completion_kind": "IDEA_YIELD",
            }
        )


def test_completion_rejects_unresolved_errors_for_construction_and_assignment() -> None:
    zero_evidence = ZeroSurvivorEvidence.from_gate(
        run_id="run-1",
        gate_stage=RunStage.QUALITY_GATED,
        input_hashes={},
    )
    with pytest.raises(ValidationError, match="unresolved errors"):
        RunState(
            run_id="run-1",
            stage=RunStage.COMPLETE,
            completion_kind=CompletionKind.AUDITABLE_ZERO_SURVIVOR,
            completion_evidence=zero_evidence,
            unresolved_errors=["timeout"],
        )

    state = RunState(run_id="run-2", stage=RunStage.HUMAN_SCORED)
    state.finalize_idea_yield([ready_pack()])

    with pytest.raises(ValueError, match="transition APIs"):
        state.unresolved_errors = ["timeout"]
    with pytest.raises(AttributeError):
        state.unresolved_errors.append("timeout")
    with pytest.raises(TypeError):
        list.append(state.unresolved_errors, "timeout")

    assert state.unresolved_errors == ()


def test_model_copy_and_revalidation_reject_completion_bypasses() -> None:
    state = RunState(run_id="run-1")

    with pytest.raises(ValidationError, match="completion_evidence"):
        state.model_copy(
            update={
                "stage": RunStage.COMPLETE,
                "completion_kind": CompletionKind.IDEA_YIELD,
            }
        )

    completed = RunState(run_id="run-2", stage=RunStage.HUMAN_SCORED)
    completed.finalize_idea_yield([ready_pack()])
    with pytest.raises(ValidationError, match="unresolved errors"):
        completed.model_copy(update={"unresolved_errors": ["timeout"]})

    with pytest.raises(ValidationError, match="unresolved errors"):
        RunState.model_construct(
            run_id="run-2",
            stage=RunStage.COMPLETE,
            completion_kind=CompletionKind.AUDITABLE_ZERO_SURVIVOR,
            completion_evidence=ZeroSurvivorEvidence.from_gate(
                run_id="run-2",
                gate_stage=RunStage.QUALITY_GATED,
                input_hashes={},
            ),
            unresolved_errors=["timeout"],
        )


def test_run_state_revalidates_constructed_nested_completion_evidence() -> None:
    invalid_evidence = IdeaYieldEvidence.model_construct(
        human_confirmed_ready_packs=()
    )

    with pytest.raises(ValidationError):
        RunState(
            run_id="run-1",
            stage=RunStage.COMPLETE,
            completion_kind=CompletionKind.IDEA_YIELD,
            completion_evidence=invalid_evidence,
        )


def test_completed_evidence_rejects_digest_count_id_and_input_tampering() -> None:
    state = RunState(
        run_id="run-1",
        stage=RunStage.HUMAN_SCORED,
        input_hashes={"corpus": "abc123"},
    )
    state.finalize_idea_yield([ready_pack()])
    payload = state.model_dump(mode="json")

    digest_tamper = deepcopy(payload)
    digest_tamper["completion_evidence"]["artifact"]["artifact_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="digest"):
        RunState.model_validate(digest_tamper)

    count_tamper = deepcopy(payload)
    count_tamper["completion_evidence"]["artifact"]["record_count"] = 2
    with pytest.raises(ValidationError, match="record_count"):
        RunState.model_validate(count_tamper)

    id_tamper = deepcopy(payload)
    id_tamper["completion_evidence"]["artifact"]["record_ids"] = [
        "idea-other"
    ]
    with pytest.raises(ValidationError, match="record_ids"):
        RunState.model_validate(id_tamper)

    input_tamper = deepcopy(payload)
    input_tamper["input_hashes"]["corpus"] = "changed"
    with pytest.raises(ValidationError, match="input hash snapshot"):
        RunState.model_validate(input_tamper)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), "idea_factory.completion_manifest.v0"),
        (("kind",), "AUDITABLE_ZERO_SURVIVOR"),
        (("run_id",), "run-other"),
        (("input_hash_snapshot",), [["corpus", "changed"]]),
        (("artifact", "artifact_ref"), "inline:other"),
        (("artifact", "artifact_sha256"), "0" * 64),
        (("artifact", "record_ids"), ["idea-other"]),
        (("artifact", "record_count"), 2),
        (("manifest_sha256",), "0" * 64),
    ],
)
def test_completion_manifest_digest_rejects_envelope_tampering(
    path: tuple[str, ...], value: object
) -> None:
    state = RunState(
        run_id="run-1",
        stage=RunStage.HUMAN_SCORED,
        input_hashes={"corpus": "abc"},
    )
    state.finalize_idea_yield([ready_pack()])
    payload = state.completion_evidence.model_dump(mode="json")
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValidationError):
        CompletionManifest.model_validate(payload)


def test_zero_survivor_completion_binds_gate_decision_and_empty_candidate_set() -> None:
    state = RunState(
        run_id="run-1",
        stage=RunStage.QUALITY_GATED,
        input_hashes={"opportunities": "sha-opportunities"},
    )

    state.advance(RunStage.COMPLETE, zero_survivor=True)

    evidence = state.completion_evidence
    assert isinstance(evidence, ZeroSurvivorEvidence)
    assert evidence.candidate_record_ids == ()
    assert evidence.candidate_count == 0
    assert evidence.decision_count == 1
    assert len(evidence.decision_record_ids) == 1
    assert evidence.gate_artifact.record_ids == evidence.decision_record_ids
    assert evidence.input_hash_snapshot == (("opportunities", "sha-opportunities"),)
    assert len(evidence.manifest_sha256) == 64
    assert evidence.manifest_sha256 != evidence.artifact.artifact_sha256


def test_artifact_evidence_revalidates_embedded_canonical_records() -> None:
    evidence = ArtifactEvidence.from_records(
        artifact_ref="inline:test-records",
        records=({"record_id": "record-1", "value": "路线"},),
    )
    payload = evidence.model_dump(mode="json")

    assert payload["record_count"] == 1
    assert payload["record_ids"] == ["record-1"]
    assert len(payload["artifact_sha256"]) == 64

    payload["canonical_records_json"] = "[]"
    with pytest.raises(ValidationError, match="record_count|digest"):
        ArtifactEvidence.model_validate(payload)


def test_idea_completion_rejects_missing_pack_record_without_raw_key_error() -> None:
    generic = ArtifactEvidence.from_records(
        artifact_ref="inline:generic",
        records=({"record_id": "record-1", "value": "not-an-idea-pack"},),
    )

    state = RunState(run_id="run-1", stage=RunStage.HUMAN_SCORED)
    state.finalize_idea_yield([ready_pack()])
    payload = state.completion_evidence.model_dump(mode="json")
    payload["artifact"] = generic.model_dump(mode="json")

    with pytest.raises((ValidationError, ValueError)):
        CompletionManifest.model_validate(payload)


def test_completed_state_round_trips_with_persisted_evidence() -> None:
    state = RunState(run_id="run-1", stage=RunStage.HUMAN_SCORED)
    state.finalize_idea_yield([ready_pack()])

    restored = RunState.model_validate(state.model_dump(mode="json"))

    assert restored.stage == RunStage.COMPLETE
    assert restored.completion_kind == CompletionKind.IDEA_YIELD
    assert isinstance(restored.completion_evidence, IdeaYieldEvidence)
    assert restored.completion_evidence.human_confirmed_ready_packs[0].idea_id == "idea-1"


@pytest.mark.parametrize(
    "data",
    [
        {"run_id": "run-1", "stage": RunStage.COMPLETE},
        {"run_id": "run-1", "completion_kind": CompletionKind.IDEA_YIELD},
        {"run_id": " "},
        {"run_id": "run-1", "input_hashes": {"": "abc"}},
        {"run_id": "run-1", "input_hashes": {"corpus": " "}},
        {"run_id": "run-1", "unresolved_errors": [" "]},
    ],
)
def test_run_state_rejects_invalid_serialized_state(data: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RunState.model_validate(data)


def test_run_state_default_containers_are_independent_and_values_nonblank() -> None:
    first = RunState(run_id="run-1")
    second = RunState(run_id="run-2")

    first.set_input_hash("corpus", "abc")
    first.add_unresolved_error("retry required")

    assert dict(second.input_hashes) == {}
    assert second.unresolved_errors == ()
    with pytest.raises(ValueError, match="transition APIs"):
        first.run_id = " "
    assert first.run_id == "run-1"


def test_run_state_rejects_direct_lifecycle_assignment_without_corrupting_state() -> None:
    state = RunState(run_id="run-1")

    with pytest.raises(ValueError):
        state.stage = RunStage.COMPLETE
    with pytest.raises(ValueError):
        state.completion_kind = CompletionKind.IDEA_YIELD

    assert state.stage == RunStage.CREATED
    assert state.completion_kind is None


def test_input_hashes_are_structurally_immutable_and_use_validated_update_api() -> None:
    state = RunState(run_id="run-1", input_hashes={"corpus": "abc"})

    with pytest.raises(ValueError, match="transition APIs"):
        state.input_hashes = {"corpus": "changed"}
    with pytest.raises(TypeError):
        state.input_hashes["corpus"] = "changed"
    with pytest.raises(TypeError):
        dict.__setitem__(state.input_hashes, "corpus", "changed")

    state.set_input_hash("corpus", "changed")

    assert dict(state.input_hashes) == {"corpus": "changed"}
    assert state.model_dump(mode="json")["input_hashes"] == {
        "corpus": "changed"
    }


def test_unresolved_errors_are_tuple_backed_with_explicit_add_resolve_api() -> None:
    state = RunState(run_id="run-1")

    state.add_unresolved_error("timeout")

    assert state.unresolved_errors == ("timeout",)
    assert state.model_dump(mode="json")["unresolved_errors"] == ["timeout"]
    with pytest.raises(AttributeError):
        state.unresolved_errors.append("other")
    with pytest.raises(TypeError):
        list.append(state.unresolved_errors, "other")

    state.resolve_unresolved_error("timeout")

    assert state.unresolved_errors == ()


def test_completed_state_provenance_cannot_be_changed_through_any_public_path() -> None:
    state = RunState(
        run_id="run-1",
        stage=RunStage.HUMAN_SCORED,
        input_hashes={"corpus": "abc"},
    )
    state.finalize_idea_yield([ready_pack()])

    with pytest.raises(ValueError, match="COMPLETE"):
        state.set_input_hash("corpus", "changed")
    with pytest.raises(ValueError, match="COMPLETE"):
        state.add_unresolved_error("timeout")
    with pytest.raises(TypeError):
        dict.__setitem__(state.input_hashes, "corpus", "changed")
    with pytest.raises(TypeError):
        list.append(state.unresolved_errors, "timeout")

    assert dict(state.input_hashes) == {"corpus": "abc"}
    assert state.unresolved_errors == ()


def test_run_id_assignment_is_rollback_safe_before_and_after_completion() -> None:
    active = RunState(run_id="run-active")
    complete = RunState(run_id="run-complete", stage=RunStage.HUMAN_SCORED)
    complete.finalize_idea_yield([ready_pack()])

    for state in (active, complete):
        original_id = state.run_id
        with pytest.raises(ValueError, match="transition APIs"):
            state.run_id = "run-mutated"
        with pytest.raises(ValueError, match="immutable"):
            state.model_copy(update={"run_id": "run-mutated"})
        assert state.run_id == original_id

    assert complete.completion_evidence.run_id == "run-complete"


def test_input_hashes_defensively_copy_mappingproxy_backing_storage() -> None:
    backing = {"corpus": "abc"}
    supplied = MappingProxyType(backing)
    state = RunState(
        run_id="run-1",
        stage=RunStage.HUMAN_SCORED,
        input_hashes=supplied,
    )

    backing["corpus"] = "mutated-before-completion"
    assert dict(state.input_hashes) == {"corpus": "abc"}

    state.finalize_idea_yield([ready_pack()])
    backing["corpus"] = "mutated-after-completion"

    assert dict(state.input_hashes) == {"corpus": "abc"}
    assert state.completion_evidence.input_hash_snapshot == (("corpus", "abc"),)


@pytest.mark.parametrize("second_value", ["aaa", "bbb"])
def test_input_hashes_reject_duplicate_normalized_keys(second_value: str) -> None:
    with pytest.raises(ValidationError, match="duplicate normalized"):
        RunState(input_hashes={"source": "aaa", " source ": second_value}, run_id="run-1")
