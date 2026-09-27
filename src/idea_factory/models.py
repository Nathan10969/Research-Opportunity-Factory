"""Strict schema contracts for the Idea Factory v1 pipeline."""

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from enum import StrEnum
import hashlib
import json
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self, Sequence

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_serializer,
    field_validator,
    model_validator,
)


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Score = Annotated[int, Field(ge=0, le=5, strict=True)]
ReconResultCount = Annotated[int, Field(ge=1, le=10, strict=True)]
NonNegativeInt = Annotated[int, Field(ge=0, strict=True)]
Sha256Hex = Annotated[
    str, StringConstraints(pattern=r"^[0-9a-f]{64}$", strict=True)
]


class StrictModel(BaseModel):
    """Base model that rejects unknown fields and validates reassignment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class FrozenStrictModel(StrictModel):
    """Strict model for evidence that must remain valid after construction."""

    model_config = ConfigDict(frozen=True, revalidate_instances="always")

    def model_copy(
        self, *, update: Mapping[str, object] | None = None, deep: bool = False
    ) -> Self:
        """Copy unchanged state directly and fully validate requested updates."""

        if update is None:
            return super().model_copy(deep=deep)
        data = {
            field_name: getattr(self, field_name)
            for field_name in self.model_fields_set
        }
        if deep:
            data = deepcopy(data)
        data.update(update)
        copied = type(self).model_validate(data)
        if not deep:
            for field_name in type(self).model_fields:
                if field_name not in update:
                    previous_value = getattr(self, field_name)
                    validated_value = getattr(copied, field_name)
                    try:
                        values_are_equal = previous_value == validated_value
                    except Exception:
                        continue
                    if (
                        type(previous_value) is type(validated_value)
                        and values_are_equal is True
                    ):
                        object.__setattr__(copied, field_name, previous_value)
        object.__setattr__(
            copied,
            "__pydantic_fields_set__",
            set(self.model_fields_set) | set(update),
        )
        return copied


class EvidenceStatus(StrEnum):
    REPORTED = "REPORTED"
    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    UNKNOWN = "UNKNOWN"


class CorpusLabel(StrEnum):
    KV_CACHE = "KV_CACHE"
    LONG_MEMORY = "LONG_MEMORY"
    BRIDGE = "BRIDGE"
    HUMAN_SUPERVISION = "HUMAN_SUPERVISION"
    SHIFT_ROBUSTNESS = "SHIFT_ROBUSTNESS"
    SUPERVISION_SHIFT_BRIDGE = "SUPERVISION_SHIFT_BRIDGE"
    GENERAL_RESEARCH = "GENERAL_RESEARCH"
    OTHER = "OTHER"


class OpportunityOperator(StrEnum):
    ASSUMPTION_BREAK = "ASSUMPTION_BREAK"
    FAILURE_TRANSFER = "FAILURE_TRANSFER"
    BOUNDARY_CONDITION = "BOUNDARY_CONDITION"
    MEASUREMENT_GAP = "MEASUREMENT_GAP"
    EVALUATION_MISMATCH = "EVALUATION_MISMATCH"
    OBJECTIVE_CONFLICT = "OBJECTIVE_CONFLICT"
    DYNAMIC_MISMATCH = "DYNAMIC_MISMATCH"
    MECHANISM_TRANSPLANT = "MECHANISM_TRANSPLANT"


class ReconDecision(StrEnum):
    COVERED = "COVERED"
    NEAR_PRIOR_WITH_RESIDUAL = "NEAR_PRIOR_WITH_RESIDUAL"
    NO_DIRECT_COVERAGE_FOUND = "NO_DIRECT_COVERAGE_FOUND"


class IdeaStatus(StrEnum):
    READY_FOR_CHEAP_TEST = "READY_FOR_CHEAP_TEST"
    HOLD = "HOLD"
    KILLED_QUALITY = "KILLED_QUALITY"
    KILLED_INTERNAL_DUP = "KILLED_INTERNAL_DUP"
    KILLED_RECON = "KILLED_RECON"
    KILLED_REVIEW = "KILLED_REVIEW"


class ReviewDecision(StrEnum):
    KILL = "KILL"
    NARROW = "NARROW"
    PASS_TO_HUMAN = "PASS_TO_HUMAN"


class EvidencePointer(StrictModel):
    source: Literal["NOTE", "PDF", "TABLE", "FIGURE", "EXPERIMENT"]
    locator: NonEmptyStr
    supports: NonEmptyStr


class FailureClaim(StrictModel):
    text: str
    status: EvidenceStatus


class PaperMeta(StrictModel):
    title: str
    year: int
    venue: str
    source_path: str


class Scope(StrictModel):
    object: str
    time_horizon: str
    setting: str


class Evaluation(StrictModel):
    measurement: str
    regime: str


class ConceptAssignment(FrozenStrictModel):
    """One auditable mechanical or reviewed concept assignment.

    ``normalized_text`` is mechanical unless an explicitly reviewed assignment
    replaces it.  The record deliberately carries its scope binding so a
    review row cannot silently transfer a concept across regimes.
    """

    schema_version: Literal["idea_factory.concept_assignment.v1"] = (
        "idea_factory.concept_assignment.v1"
    )
    dimension: NonEmptyStr
    facet: NonEmptyStr
    raw_text: NonEmptyStr
    normalized_text: NonEmptyStr
    card_id: NonEmptyStr
    scope_key: NonEmptyStr
    merge_reason: str = ""
    disputed: bool = False
    reviewed: bool = False
    review_status: Literal["MECHANICAL", "APPROVED", "DISPUTED"] = "MECHANICAL"
    reviewer_reason: str = ""


class ReviewedClusterScope(FrozenStrictModel):
    """Nonblank canonical scope supplied by a human ontology review."""

    object: NonEmptyStr
    time_horizon: NonEmptyStr
    setting: NonEmptyStr


class ReviewedConceptAssignmentV2(FrozenStrictModel):
    """Human-reviewed concept and canonical cluster-scope assignment.

    ``source_scope_key`` binds the row to the exact evidence scope on the
    PaperCard. ``cluster_scope`` is a separate reviewed ontology projection;
    it never rewrites the source scope.
    """

    schema_version: Literal["idea_factory.concept_assignment.v2"]
    dimension: NonEmptyStr
    facet: NonEmptyStr
    raw_text: NonEmptyStr
    normalized_text: NonEmptyStr
    card_id: NonEmptyStr
    source_scope_key: NonEmptyStr
    cluster_scope: ReviewedClusterScope
    merge_reason: NonEmptyStr
    disputed: bool
    reviewed: bool
    review_status: Literal["APPROVED", "DISPUTED"]
    reviewer_reason: NonEmptyStr


class PaperCard(StrictModel):
    schema_version: Literal["idea_factory.paper_card.v1"]
    card_id: str
    paper: PaperMeta
    problem: str
    assumption: FailureClaim
    mechanism: str
    failure_observation: FailureClaim
    failure_mechanism: FailureClaim
    limitation: str
    evaluation: Evaluation
    scope: Scope
    evidence_pointers: list[EvidencePointer]
    extraction_confidence: Literal["HIGH", "MEDIUM", "LOW"]


class Opportunity(StrictModel):
    schema_version: Literal["idea_factory.opportunity.v1"]
    opportunity_id: str
    operator: OpportunityOperator
    assumption_x: str
    observation_y: str
    condition_z: str
    failure_f: str
    missing_capability_w: str
    alternative_explanation_a: str
    decisive_experiment: str
    supporting_card_ids: list[str]
    nearest_internal_neighbors: list[str]
    scope_compatibility: str
    inference_flags: list[str] = Field(default_factory=list)


class QualityDecision(StrictModel):
    opportunity_id: str
    passed: bool
    reason_codes: list[str]


class ReconQuery(StrictModel):
    query_id: str
    opportunity_id: str
    lane: Literal["CONCEPT", "MECHANISM", "FAILURE", "EVALUATION"]
    variant: Literal["CURRENT_TERMS", "GENERIC_SHAPE", "SYNONYM"]
    query: str
    max_results: ReconResultCount


class NearestPrior(FrozenStrictModel):
    paper: NonEmptyStr
    exact_overlap: NonEmptyStr
    residual_difference: NonEmptyStr
    evidence_url_or_id: NonEmptyStr


class ReconReport(StrictModel):
    schema_version: Literal["idea_factory.recon.v1"]
    opportunity_id: str
    searched_query_ids: list[str]
    searched_at: datetime
    nearest_priors: list[NearestPrior]
    decision: ReconDecision
    decision_reason: str


class DecisiveTest(FrozenStrictModel):
    setup: NonEmptyStr
    discriminates_against: NonEmptyStr
    expected_runtime_or_cost: NonEmptyStr


class RouteProposal(StrictModel):
    route_id: str
    opportunity_id: str
    old_assumption_changed: str
    new_assumption: str
    new_mechanism: str
    why_it_addresses_failure: str
    why_nearest_priors_cannot: str
    source_of_gain: str
    cheapest_decisive_test: DecisiveTest
    kill_condition: str


class ReviewVerdict(StrictModel):
    route_id: str
    decision: ReviewDecision
    strongest_baseline: str
    a_plus_b_objection: str
    source_of_gain_verdict: str
    falsifiability_verdict: str
    cost_risk: str
    residual_claim: str


class HumanScores(FrozenStrictModel):
    specific_novelty: Score
    importance: Score
    paper_potential: Score
    feasibility: Score
    excitement: Score
    evidence_clarity: Score


class IdeaPack(FrozenStrictModel):
    schema_version: Literal["idea_factory.idea_pack.v1"] = "idea_factory.idea_pack.v1"
    idea_id: str
    status: IdeaStatus
    opportunity: str
    core_hypothesis: str
    why_now: str
    supporting_observations: tuple[str, ...] = ()
    inference_flags: tuple[str, ...] = ()
    nearest_priors: tuple[NearestPrior, ...] = ()
    proposed_mechanism: str
    source_of_gain: str
    cheapest_decisive_test: DecisiveTest | None = None
    strongest_baseline: str
    kill_condition: str
    main_uncertainty: str
    expected_reviewer_2_objection: str
    response_to_objection: str
    evidence_that_would_make_reviewer_correct: str
    human_scores: HumanScores | None = None
    human_reason_codes: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def require_readiness_evidence(self) -> Self:
        if self.status != IdeaStatus.READY_FOR_CHEAP_TEST:
            return self

        missing: list[str] = []
        for field_name in ("source_of_gain", "strongest_baseline", "kill_condition"):
            if not getattr(self, field_name).strip():
                missing.append(field_name)
        if not self.nearest_priors:
            missing.append("nearest_priors")
        if self.cheapest_decisive_test is None:
            missing.append("cheapest_decisive_test")
        if self.human_scores is None:
            missing.append("human_scores")
        if not self.human_reason_codes:
            missing.append("human_reason_codes")
        if missing:
            raise ValueError(
                "READY_FOR_CHEAP_TEST requires: " + ", ".join(missing)
            )
        return self


class RunStage(StrEnum):
    CREATED = "CREATED"
    CORPUS_ROUTED = "CORPUS_ROUTED"
    CARDS_READY = "CARDS_READY"
    LANDSCAPE_READY = "LANDSCAPE_READY"
    OPPORTUNITIES_READY = "OPPORTUNITIES_READY"
    QUALITY_GATED = "QUALITY_GATED"
    RECON_READY = "RECON_READY"
    ROUTES_READY = "ROUTES_READY"
    REVIEWED = "REVIEWED"
    HUMAN_SCORED = "HUMAN_SCORED"
    COMPLETE = "COMPLETE"


class CompletionKind(StrEnum):
    IDEA_YIELD = "IDEA_YIELD"
    AUDITABLE_ZERO_SURVIVOR = "AUDITABLE_ZERO_SURVIVOR"


RUN_ORDER = (
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

ZERO_SURVIVOR_STAGES = (
    RunStage.QUALITY_GATED,
    RunStage.RECON_READY,
    RunStage.REVIEWED,
)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value} is forbidden")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


ImmutableStringMap = type(MappingProxyType({}))


def _freeze_string_map(values: Mapping[str, str]) -> ImmutableStringMap:
    normalized: dict[str, str] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("input hash keys must be nonblank strings")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("input hash values must be nonblank strings")
        normalized_key = key.strip()
        if normalized_key in normalized:
            raise ValueError(f"duplicate normalized input hash key: {normalized_key}")
        normalized[normalized_key] = value.strip()
    return MappingProxyType(normalized)


InputHashSnapshot = tuple[tuple[NonEmptyStr, NonEmptyStr], ...]


def _input_hash_snapshot(values: Mapping[str, str]) -> InputHashSnapshot:
    return tuple(sorted(values.items()))


class ArtifactEvidence(FrozenStrictModel):
    """Content-addressed assertion over embedded canonical artifact records."""

    artifact_ref: NonEmptyStr
    artifact_sha256: Sha256Hex
    record_ids: tuple[NonEmptyStr, ...]
    record_count: NonNegativeInt
    canonical_records_json: NonEmptyStr

    @classmethod
    def from_records(
        cls,
        *,
        artifact_ref: str,
        records: Sequence[Mapping[str, Any]],
    ) -> "ArtifactEvidence":
        record_values = [dict(record) for record in records]
        canonical_records_json = _canonical_json(record_values)
        record_ids = tuple(record.get("record_id") for record in record_values)
        return cls(
            artifact_ref=artifact_ref,
            artifact_sha256=_sha256_text(canonical_records_json),
            record_ids=record_ids,
            record_count=len(record_values),
            canonical_records_json=canonical_records_json,
        )

    @model_validator(mode="after")
    def verify_content_binding(self) -> Self:
        records = _parse_json_records(self.canonical_records_json)
        canonical = _canonical_json(records)
        if canonical != self.canonical_records_json:
            raise ValueError("artifact records must use canonical JSON")
        if _sha256_text(canonical) != self.artifact_sha256:
            raise ValueError("artifact digest does not match embedded records")
        if self.record_count != len(records):
            raise ValueError("artifact record_count does not match embedded records")
        embedded_ids = tuple(record.get("record_id") for record in records)
        if embedded_ids != self.record_ids:
            raise ValueError("artifact record_ids do not match embedded records")
        if len(set(self.record_ids)) != len(self.record_ids):
            raise ValueError("artifact record_ids must be unique")
        return self

    def records(self) -> tuple[dict[str, Any], ...]:
        return _parse_json_records(self.canonical_records_json)


def _parse_json_records(value: str) -> tuple[dict[str, Any], ...]:
    try:
        records = json.loads(value, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("artifact records must be strict JSON") from exc
    if not isinstance(records, list) or any(
        not isinstance(record, dict) for record in records
    ):
        raise ValueError("artifact records must be a JSON array of objects")
    return tuple(records)


def _manifest_envelope(
    *,
    schema_version: str,
    kind: CompletionKind,
    run_id: str,
    input_hash_snapshot: InputHashSnapshot,
    artifact: ArtifactEvidence,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "completion_kind": kind.value,
        "run_id": run_id,
        "input_hash_snapshot": input_hash_snapshot,
        "artifact_ref": artifact.artifact_ref,
        "artifact_sha256": artifact.artifact_sha256,
        "record_ids": artifact.record_ids,
        "record_count": artifact.record_count,
    }


def _idea_records(
    packs: Sequence[IdeaPack], snapshot: InputHashSnapshot
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "record_id": pack.idea_id,
            "input_hash_snapshot": snapshot,
            "idea_pack": pack.model_dump(mode="json"),
            "human_attestation": {
                "scores": pack.human_scores.model_dump(mode="json"),
                "reason_codes": list(pack.human_reason_codes),
            },
        }
        for pack in packs
        if pack.human_scores is not None
    )


class CompletionManifest(FrozenStrictModel):
    """Content-addressed completion assertion, fully checkable on reload."""

    schema_version: Literal["idea_factory.completion_manifest.v1"] = (
        "idea_factory.completion_manifest.v1"
    )
    kind: CompletionKind
    run_id: NonEmptyStr
    input_hash_snapshot: InputHashSnapshot
    artifact: ArtifactEvidence
    manifest_sha256: Sha256Hex

    @classmethod
    def _from_artifact(
        cls,
        *,
        kind: CompletionKind,
        run_id: str,
        input_hash_snapshot: InputHashSnapshot,
        artifact: ArtifactEvidence,
    ) -> "CompletionManifest":
        schema_version = "idea_factory.completion_manifest.v1"
        envelope = _manifest_envelope(
            schema_version=schema_version,
            kind=kind,
            run_id=run_id,
            input_hash_snapshot=input_hash_snapshot,
            artifact=artifact,
        )
        return cls(
            schema_version=schema_version,
            kind=kind,
            run_id=run_id,
            input_hash_snapshot=input_hash_snapshot,
            artifact=artifact,
            manifest_sha256=_sha256_text(_canonical_json(envelope)),
        )

    @classmethod
    def from_ready_packs(
        cls, *, run_id: str, input_hashes: Mapping[str, str], packs: Sequence[IdeaPack]
    ) -> "CompletionManifest":
        ready = tuple(IdeaPack.model_validate(pack.model_dump()) for pack in packs)
        snapshot = _input_hash_snapshot(input_hashes)
        artifact = ArtifactEvidence.from_records(
            artifact_ref=f"inline:run-state/{run_id}/ready-packs-with-attestations",
            records=_idea_records(ready, snapshot),
        )
        return cls._from_artifact(
            kind=CompletionKind.IDEA_YIELD,
            run_id=run_id,
            input_hash_snapshot=snapshot,
            artifact=artifact,
        )

    @classmethod
    def from_gate(
        cls, *, run_id: str, gate_stage: RunStage, input_hashes: Mapping[str, str]
    ) -> "CompletionManifest":
        if gate_stage not in ZERO_SURVIVOR_STAGES:
            raise ValueError("zero-survivor evidence requires an allowed gate stage")
        snapshot = _input_hash_snapshot(input_hashes)
        body: dict[str, Any] = {
            "run_id": run_id,
            "gate_stage": gate_stage.value,
            "input_hash_snapshot": snapshot,
            "candidate_record_ids": (),
            "candidate_count": 0,
            "decision": CompletionKind.AUDITABLE_ZERO_SURVIVOR.value,
        }
        decision_ids = (f"gate-{_sha256_text(_canonical_json(body))[:16]}",)
        artifact = ArtifactEvidence.from_records(
            artifact_ref=f"inline:run-state/{run_id}/{gate_stage.value}/decisions",
            records=(
                {
                    "record_id": decision_ids[0],
                    "decision_record_ids": decision_ids,
                    "decision_count": 1,
                }
                | body,
            ),
        )
        return cls._from_artifact(
            kind=CompletionKind.AUDITABLE_ZERO_SURVIVOR,
            run_id=run_id,
            input_hash_snapshot=snapshot,
            artifact=artifact,
        )

    @model_validator(mode="after")
    def verify_bindings(self) -> Self:
        if self.input_hash_snapshot != tuple(sorted(self.input_hash_snapshot)):
            raise ValueError("input hash snapshot must be sorted")
        if len({name for name, _ in self.input_hash_snapshot}) != len(
            self.input_hash_snapshot
        ):
            raise ValueError("input hash snapshot keys must be unique")
        if self.kind == CompletionKind.IDEA_YIELD:
            records = self.artifact.records()
            if any("idea_pack" not in record for record in records):
                raise ValueError("IDEA_YIELD artifact records require idea_pack")
            packs = tuple(
                IdeaPack.model_validate(record["idea_pack"])
                for record in records
            )
            if not packs or any(
                pack.status != IdeaStatus.READY_FOR_CHEAP_TEST
                or pack.human_scores is None
                or not pack.human_reason_codes
                for pack in packs
            ):
                raise ValueError("IDEA_YIELD requires human-confirmed READY IdeaPacks")
            expected_records = _idea_records(packs, self.input_hash_snapshot)
        else:
            records = self.artifact.records()
            if len(records) != 1:
                raise ValueError("invalid zero-survivor gate manifest")
            record = records[0]
            gate_stage = RunStage(record.get("gate_stage"))
            body = {
                "run_id": self.run_id,
                "gate_stage": gate_stage.value,
                "input_hash_snapshot": self.input_hash_snapshot,
                "candidate_record_ids": (),
                "candidate_count": 0,
                "decision": self.kind.value,
            }
            decision_id = f"gate-{_sha256_text(_canonical_json(body))[:16]}"
            expected_records = (
                {
                    "record_id": decision_id,
                    "decision_record_ids": (decision_id,),
                    "decision_count": 1,
                }
                | body,
            )
            if gate_stage not in ZERO_SURVIVOR_STAGES:
                raise ValueError("invalid zero-survivor gate manifest")
        expected = ArtifactEvidence.from_records(
            artifact_ref=self.artifact.artifact_ref, records=expected_records
        )
        if self.artifact != expected:
            raise ValueError("completion artifact binding does not match records")
        envelope = _manifest_envelope(
            schema_version=self.schema_version,
            kind=self.kind,
            run_id=self.run_id,
            input_hash_snapshot=self.input_hash_snapshot,
            artifact=self.artifact,
        )
        if _sha256_text(_canonical_json(envelope)) != self.manifest_sha256:
            raise ValueError("completion manifest digest does not match envelope")
        return self

    @property
    def ready_pack_artifact(self) -> ArtifactEvidence:
        return self.artifact

    @property
    def human_confirmed_ready_packs(self) -> tuple[IdeaPack, ...]:
        return tuple(
            IdeaPack.model_validate(record["idea_pack"])
            for record in self.ready_pack_artifact.records()
        )

    @property
    def human_attestation_artifact(self) -> ArtifactEvidence:
        return self.artifact

    @property
    def gate_artifact(self) -> ArtifactEvidence:
        return self.artifact

    @property
    def gate_stage(self) -> RunStage:
        return RunStage(self.artifact.records()[0]["gate_stage"])

    @property
    def candidate_record_ids(self) -> tuple[str, ...]:
        return tuple(self.artifact.records()[0].get("candidate_record_ids", ()))

    @property
    def candidate_count(self) -> int:
        return int(self.artifact.records()[0].get("candidate_count", 0))

    @property
    def decision_record_ids(self) -> tuple[str, ...]:
        return tuple(self.artifact.records()[0].get("decision_record_ids", ()))

    @property
    def decision_count(self) -> int:
        return int(self.artifact.records()[0].get("decision_count", 0))


IdeaYieldEvidence = CompletionManifest
ZeroSurvivorEvidence = CompletionManifest


class RunState(StrictModel):
    """Validated state machine for one reproducible Idea Factory run."""

    run_id: NonEmptyStr
    stage: RunStage = RunStage.CREATED
    input_hashes: ImmutableStringMap = Field(
        default_factory=lambda: MappingProxyType({})
    )
    completion_kind: CompletionKind | None = None
    completion_evidence: CompletionManifest | None = None
    unresolved_errors: tuple[NonEmptyStr, ...] = ()

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_assignment=True,
        revalidate_instances="always",
    )

    @field_validator("input_hashes", mode="before")
    @classmethod
    def freeze_input_hashes(cls, value: object) -> ImmutableStringMap:
        if not isinstance(value, Mapping):
            raise ValueError("input_hashes must be an object")
        return _freeze_string_map(value)

    @field_serializer("input_hashes")
    def serialize_input_hashes(self, value: ImmutableStringMap) -> dict[str, str]:
        return dict(value)

    @model_validator(mode="after")
    def validate_completion_state(self) -> Self:
        if self.stage == RunStage.COMPLETE:
            if self.completion_kind is None:
                raise ValueError("COMPLETE run state requires completion_kind")
            if self.unresolved_errors:
                raise ValueError("COMPLETE run state forbids unresolved errors")
            if (
                not isinstance(self.completion_evidence, CompletionManifest)
                or self.completion_evidence.kind != self.completion_kind
            ):
                raise ValueError("completion_kind requires matching completion_evidence")
            if self.completion_evidence.run_id != self.run_id:
                raise ValueError("completion evidence run_id does not match run state")
            if self.completion_evidence.input_hash_snapshot != _input_hash_snapshot(
                self.input_hashes
            ):
                raise ValueError(
                    "completion evidence input hash snapshot does not match run state"
                )
        elif self.completion_kind is not None or self.completion_evidence is not None:
            raise ValueError("non-COMPLETE run state forbids completion metadata")
        return self

    def __setattr__(self, name: str, value: object) -> None:
        """Keep lifecycle mutations inside the transition APIs.

        Pydantic's assignment validation can raise after an ``after`` validator
        has observed a new value, leaving the object contradictory.  Lifecycle
        fields are therefore changed only after this class has validated a
        complete replacement state.
        """

        if name in {
            "run_id",
            "stage",
            "completion_kind",
            "completion_evidence",
            "input_hashes",
            "unresolved_errors",
        } and hasattr(self, "run_id"):
            raise ValueError(f"assign {name} through RunState transition APIs")
        super().__setattr__(name, value)

    @classmethod
    def model_construct(
        cls, _fields_set: set[str] | None = None, **values: object
    ) -> Self:
        """Disallow Pydantic's unchecked construction for lifecycle state."""

        return cls.model_validate(values)

    def model_copy(
        self, *, update: Mapping[str, object] | None = None, deep: bool = False
    ) -> Self:
        """Copy through full validation so completion evidence cannot be bypassed."""

        if update and update.get("run_id", self.run_id) != self.run_id:
            raise ValueError("run_id is immutable after initial construction")
        data = self.model_dump()
        if update:
            data.update(update)
        return type(self).model_validate(data)

    def set_input_hash(self, name: str, digest: str) -> None:
        """Validate and replace one input provenance binding before completion."""

        if self.stage == RunStage.COMPLETE:
            raise ValueError("COMPLETE run provenance cannot be changed")
        updated = dict(self.input_hashes)
        updated[name] = digest
        validated = type(self).model_validate(
            self.model_dump() | {"input_hashes": updated}
        )
        object.__setattr__(self, "input_hashes", validated.input_hashes)

    def add_unresolved_error(self, error: str) -> None:
        """Add one validated unresolved error before completion."""

        if self.stage == RunStage.COMPLETE:
            raise ValueError("COMPLETE run cannot acquire unresolved errors")
        validated = type(self).model_validate(
            self.model_dump()
            | {"unresolved_errors": self.unresolved_errors + (error,)}
        )
        object.__setattr__(self, "unresolved_errors", validated.unresolved_errors)

    def resolve_unresolved_error(self, error: str) -> None:
        """Resolve all exact occurrences of an error before completion."""

        if self.stage == RunStage.COMPLETE:
            raise ValueError("COMPLETE run has no unresolved errors to change")
        normalized = error.strip() if isinstance(error, str) else error
        if normalized not in self.unresolved_errors:
            raise ValueError("unresolved error was not recorded")
        remaining = tuple(
            recorded for recorded in self.unresolved_errors if recorded != normalized
        )
        validated = type(self).model_validate(
            self.model_dump() | {"unresolved_errors": remaining}
        )
        object.__setattr__(self, "unresolved_errors", validated.unresolved_errors)

    def advance(self, target: RunStage, *, zero_survivor: bool = False) -> None:
        """Advance exactly one normal stage, or close an audited zero-survivor run."""

        if self.stage == RunStage.COMPLETE:
            raise ValueError("run is already COMPLETE and cannot advance")
        if zero_survivor:
            if self.unresolved_errors:
                raise ValueError("unresolved errors block completion")
            if target != RunStage.COMPLETE or self.stage not in ZERO_SURVIVOR_STAGES:
                raise ValueError("zero-survivor completion is only allowed from its gate stages")
            self._complete(
                CompletionKind.AUDITABLE_ZERO_SURVIVOR,
                ZeroSurvivorEvidence.from_gate(
                    run_id=self.run_id,
                    gate_stage=self.stage,
                    input_hashes=self.input_hashes,
                ),
            )
            return
        if target == RunStage.COMPLETE:
            raise ValueError("normal completion requires finalize_idea_yield with human-confirmed READY IdeaPack")
        current_index = RUN_ORDER.index(self.stage)
        if target != RUN_ORDER[current_index + 1]:
            raise ValueError("run stages must advance to the exact next stage")
        validated = type(self).model_validate(self.model_dump() | {"stage": target})
        object.__setattr__(self, "stage", validated.stage)

    def finalize_idea_yield(self, idea_packs: Sequence[IdeaPack]) -> None:
        """Complete only after at least one human-confirmed ready IdeaPack."""

        if self.stage != RunStage.HUMAN_SCORED:
            raise ValueError("IDEA_YIELD completion requires HUMAN_SCORED stage")
        if self.unresolved_errors:
            raise ValueError("unresolved errors block completion")
        try:
            evidence = IdeaYieldEvidence.from_ready_packs(
                run_id=self.run_id,
                input_hashes=self.input_hashes,
                packs=tuple(idea_packs),
            )
        except ValueError as exc:
            raise ValueError(
                "IDEA_YIELD requires at least one human-confirmed READY IdeaPack"
            ) from exc
        self._complete(CompletionKind.IDEA_YIELD, evidence)

    def _complete(
        self,
        completion_kind: CompletionKind,
        completion_evidence: CompletionManifest,
    ) -> None:
        validated = type(self).model_validate(
            self.model_dump()
            | {
                "stage": RunStage.COMPLETE,
                "completion_kind": completion_kind,
                "completion_evidence": completion_evidence,
            }
        )
        object.__setattr__(self, "stage", validated.stage)
        object.__setattr__(self, "completion_kind", validated.completion_kind)
        object.__setattr__(self, "completion_evidence", validated.completion_evidence)
