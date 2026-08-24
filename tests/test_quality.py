import pytest

from idea_factory.models import Opportunity, PaperCard


def _card(card_id: str, *, failure: str = "cache eviction causes retrieval loss") -> PaperCard:
    return PaperCard.model_validate({
        "schema_version": "idea_factory.paper_card.v1", "card_id": card_id,
        "paper": {"title": card_id, "year": 2025, "venue": "Test", "source_path": f"/notes/{card_id}.md"},
        "problem": "long context retrieval", "assumption": {"text": "cache remains resident", "status": "REPORTED"},
        "mechanism": "cache reuse", "failure_observation": {"text": "retrieval degrades after eviction", "status": "OBSERVED"},
        "failure_mechanism": {"text": failure, "status": "OBSERVED" if failure else "UNKNOWN"},
        "limitation": "does not measure eviction", "evaluation": {"measurement": "recall", "regime": "long traces"},
        "scope": {"object": "KV cache", "time_horizon": "long", "setting": "serving"},
        "evidence_pointers": [{"source": "NOTE", "locator": f"/notes/{card_id}.md#1", "supports": "assumption,failure_observation,failure_mechanism"}],
        "extraction_confidence": "HIGH",
    })


def _opportunity(**updates: object) -> Opportunity:
    body = {
        "schema_version": "idea_factory.opportunity.v1", "opportunity_id": "opp-1", "operator": "ASSUMPTION_BREAK",
        "assumption_x": "The cache remains resident during long serving traces.",
        "observation_y": "card-1 reports retrieval degrades after eviction.",
        "condition_z": "under long serving traces", "failure_f": "cache eviction causes retrieval loss",
        "missing_capability_w": "eviction-aware retrieval control", "alternative_explanation_a": "cache eviction, not routing error",
        "decisive_experiment": "Compare recall after forced cache eviction against cache eviction, not routing error.",
        "supporting_card_ids": ["card-1"], "nearest_internal_neighbors": ["card-2"],
        "scope_compatibility": "Both cards cover KV cache serving over long traces.", "inference_flags": [],
    }
    body.update(updates)
    return Opportunity.model_validate(body)


def test_sharp_opportunity_passes_deterministic_quality_gate() -> None:
    from idea_factory.quality import evaluate_opportunity
    assert evaluate_opportunity(_opportunity(), {"card-1": _card("card-1"), "card-2": _card("card-2")}).passed


def test_generic_keyword_stack_has_required_reasons() -> None:
    from idea_factory.quality import evaluate_opportunity
    decision = evaluate_opportunity(_opportunity(
        assumption_x="memory + router", observation_y="memory and router are useful", condition_z="general", failure_f="memory router issue",
        missing_capability_w="memory router", alternative_explanation_a="memory route", decisive_experiment="test memory router",
    ), {"card-1": _card("card-1"), "card-2": _card("card-2")})
    assert decision.reason_codes == ["KEYWORD_COMBINATION", "NO_DISCRIMINATING_TEST", "VAGUE_ASSUMPTION"]


def test_unknown_support_and_unknown_failure_are_rejected() -> None:
    from idea_factory.quality import evaluate_opportunity
    decision = evaluate_opportunity(_opportunity(supporting_card_ids=["missing"]), {"card-1": _card("card-1"), "card-2": _card("card-2", failure="")})
    assert "UNKNOWN_SUPPORTING_CARD" in decision.reason_codes
    unknown_failure = evaluate_opportunity(_opportunity(), {"card-1": _card("card-1", failure=""), "card-2": _card("card-2")})
    assert "UNKNOWN_FAILURE_MECHANISM" in unknown_failure.reason_codes


@pytest.mark.parametrize(("field", "reason"), [
    ("assumption_x", "EMPTY_ASSUMPTION_X"), ("observation_y", "EMPTY_OBSERVATION_Y"),
    ("condition_z", "EMPTY_CONDITION_Z"), ("failure_f", "EMPTY_FAILURE_F"),
    ("missing_capability_w", "EMPTY_MISSING_CAPABILITY_W"), ("alternative_explanation_a", "EMPTY_ALTERNATIVE_A"),
    ("decisive_experiment", "EMPTY_DECISIVE_EXPERIMENT"),
])
def test_each_blank_required_field_has_a_stable_reason(field: str, reason: str) -> None:
    from idea_factory.quality import evaluate_opportunity
    assert reason in evaluate_opportunity(_opportunity(**{field: " "}), {"card-1": _card("card-1"), "card-2": _card("card-2")}).reason_codes


def test_scope_neighbor_and_alternative_failures_are_explicit() -> None:
    from idea_factory.quality import evaluate_opportunity
    cards = {"card-1": _card("card-1"), "card-2": _card("card-2")}
    assert "UNKNOWN_INTERNAL_NEIGHBOR" in evaluate_opportunity(_opportunity(nearest_internal_neighbors=["gone"]), cards).reason_codes
    assert "BLANK_SCOPE_COMPATIBILITY" in evaluate_opportunity(_opportunity(scope_compatibility=" "), cards).reason_codes
    assert "SCOPE_CONTRADICTION" in evaluate_opportunity(_opportunity(scope_compatibility="The regimes are incompatible."), cards).reason_codes
    assert "ALTERNATIVE_NOT_NAMED" in evaluate_opportunity(_opportunity(decisive_experiment="Compare recall against the baseline."), cards).reason_codes
