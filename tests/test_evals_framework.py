"""Tests for the eval framework itself.

The framework gates the agent; nothing was gating the framework. Every bug found
in this layer so far was found by running it against production — a purge that
swallowed an HTTP 422 and reported success, a report that keyed by example and
silently discarded two thirds of a repetitions run, an empty-window KeyError
that failed every quiet hour, an agent-agnostic module with one agent's name
hardcoded in it. All four are checkable in milliseconds. None were caught before
they shipped.

Hermetic by construction: no Phoenix, no model, no network. Scores are small
synthetic DataFrames in exactly the shape `monitor.score` and
`monitor.read_annotations` produce.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pandas as pd
import pytest

# Registration is an import side effect: these modules populate REGISTRY. Without
# them every direction falls back to "maximize" and the minimize metrics score
# backwards — which is exactly what the direction tests below exist to catch, so
# they must be imported for the tests to be testing anything.
from evals.agents.travel import evaluators, judges  # noqa: F401
from evals.agents.travel import truth as T
from evals.core import config as agent_config
from evals.core import curate, diagnose, monitor, thresholds
from evals.core.config import AgentConfig
from evals.core.registry import REGISTRY, Registry

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def scores(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """(span_id, evaluator, score) -> the frame the monitor produces."""
    return pd.DataFrame(
        [{"span_id": s, "annotation_name": n, "score": v,
          "label": None, "explanation": f"because {n} said so"} for s, n, v in rows]
    )


def turn(span_id: str, message: str, calls: list[dict] | None = None, reply: str = "ok",
         turn_index: int | None = None) -> dict:
    return {
        "span_id": span_id,
        "input": {"message": message, "turn_index": turn_index},
        "output": {"reply": reply, "tool_calls": calls or []},
    }


def config(
    *,
    monitoring: dict | None = None,
    thresholds: dict | None = None,
    alerting: dict | None = None,
    curation: dict | None = None,
) -> AgentConfig:
    """A small config with one maximize and one minimize threshold."""
    return AgentConfig(
        agent="travel",
        project="p",
        dataset="d",
        agent_span_name="travel_agent",
        monitoring=monitoring if monitoring is not None else {"min_sample": 3},
        thresholds=thresholds if thresholds is not None else {
            "flight_direction": {"min_rate": 0.99},   # maximize
            "hallucination": {"max_rate": 0.05},      # minimize
        },
        alerting=alerting if alerting is not None else {
            "owners": ["eng"], "page_on": ["hallucination"],
        },
        curation=curation if curation is not None else {"enabled": True, "max_per_run": 25},
    )


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

def test_registry_filters_are_disjoint_and_complete() -> None:
    all_ = REGISTRY.for_agent("travel")
    code = REGISTRY.for_agent("travel", kind="code")
    llm = REGISTRY.for_agent("travel", kind="llm")
    assert len(code) + len(llm) == len(all_)
    assert {r.name for r in code} & {r.name for r in llm} == set()


def test_only_label_free_evaluators_are_marked_online() -> None:
    """The online set is what can run against production, where no labels exist."""
    online = {r.name for r in REGISTRY.for_agent("travel", online=True)}
    offline = {r.name for r in REGISTRY.for_agent("travel", online=False)}
    assert "tool_selection" in offline, "needs expected_tool; cannot run on live traffic"
    assert "tool_result_exact" in offline, "compares against fixtures' expected ids"
    assert {"hallucination", "graceful_alternative"} <= online, "judges need no labels"
    assert online & offline == set()


def test_registry_isolates_agents() -> None:
    r = Registry()
    r.register(agent="a", name="x", kind="code", mode="invariant")(object())
    r.register(agent="b", name="y", kind="code", mode="invariant")(object())
    assert [e.name for e in r.for_agent("a")] == ["x"]
    assert r.agents() == ["a", "b"]


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def test_discover_finds_the_travel_agent() -> None:
    agents = {c.agent for c in agent_config.discover()}
    assert "travel" in agents


def test_config_defaults_when_sections_are_absent() -> None:
    c = AgentConfig(agent="a", project="p", dataset="d", agent_span_name="s")
    assert c.schedule == "@hourly"
    assert c.min_sample == 20
    assert c.judge_max_turns is None  # uncapped unless a config asks for it
    assert c.thresholds == {}


# --------------------------------------------------------------------------
# judge sampling — the cost ceiling must never weaken an invariant
#
# These use a stub registry rather than the travel one. The behaviour under
# test is which records reach which evaluator, and running it against the real
# judges would bill four model calls per turn to assert a routing decision.
# --------------------------------------------------------------------------

class _Stub:
    """Minimal evaluator: records what it was given, returns a fixed result."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def evaluate(self, row: dict) -> list[SimpleNamespace]:
        self.seen.append(row["input"]["span_id"])
        return [SimpleNamespace(score=1.0, label="pass", explanation="stub")]


def _sampling_fixture() -> tuple[Registry, dict[str, _Stub]]:
    reg = Registry()
    stubs = {
        "code_a": _Stub(), "code_b": _Stub(),
        "judge_a": _Stub(), "judge_b": _Stub(),
    }
    for name, stub in stubs.items():
        reg.register(agent="t", name=name, mode="invariant", online=True,
                     kind="code" if name.startswith("code") else "llm")(stub)
    return reg, stubs


def _turns(n: int) -> list[dict]:
    return [{"span_id": f"s{i}", "input": {"span_id": f"s{i}"}, "output": {}}
            for i in range(n)]


def test_judge_cap_never_thins_the_code_evaluators() -> None:
    """Sampling is a cost control on the judges, not a coverage decision.

    `no_unredacted_pii` scored on a sample is not a privacy control — a leak in
    an unsampled turn is still a leak. So the cap applies to LLM evaluators
    only, and every code evaluator still sees every turn.
    """
    reg, stubs = _sampling_fixture()
    monitor.score(_turns(40), "t", judge_max_turns=5,
                  rng=random.Random(0), registry=reg)
    assert len(stubs["code_a"].seen) == 40
    assert len(stubs["code_b"].seen) == 40


def test_judge_cap_is_a_ceiling_not_a_rate() -> None:
    """Below the cap nothing is sampled, so thin traffic never gets thinner.

    This is what keeps the cap from feeding the `min_sample` floor: a sweep of
    30 turns hands all 30 to the judges, not a fraction of them.
    """
    reg, stubs = _sampling_fixture()
    monitor.score(_turns(30), "t", judge_max_turns=150,
                  rng=random.Random(0), registry=reg)
    assert len(stubs["judge_a"].seen) == 30


def test_judge_cap_draws_one_sample_shared_by_every_judge() -> None:
    """Above the ceiling the judges see exactly the cap — and the same turns.

    Sampling per judge would have each grading a different window, so their
    rates could not be compared to each other or to a previous sweep.
    """
    reg, stubs = _sampling_fixture()
    monitor.score(_turns(40), "t", judge_max_turns=5,
                  rng=random.Random(0), registry=reg)
    assert len(stubs["judge_a"].seen) == 5
    assert set(stubs["judge_a"].seen) == set(stubs["judge_b"].seen)


# --------------------------------------------------------------------------
# thresholds
# --------------------------------------------------------------------------

def test_maximize_breaches_below_its_floor() -> None:
    df = scores([(f"s{i}", "flight_direction", 0.0 if i < 2 else 1.0) for i in range(4)])
    report = thresholds.evaluate(df, config())
    assert [b.evaluator for b in report.breaches] == ["flight_direction"]
    assert report.breaches[0].rate == pytest.approx(0.5)


def test_minimize_breaches_above_its_ceiling_not_below() -> None:
    """`hallucination` scores 1.0 for the BAD outcome.

    Reading it like a maximize metric inverts the dashboard, so this pins the
    direction in both directions.
    """
    bad = scores([(f"s{i}", "hallucination", 1.0 if i < 3 else 0.0) for i in range(4)])
    assert thresholds.evaluate(bad, config()).breaches, "75% flagged must breach a 5% ceiling"

    good = scores([(f"s{i}", "hallucination", 0.0) for i in range(4)])
    assert not thresholds.evaluate(good, config()).breaches, "0% flagged must not breach"


def test_thin_samples_are_skipped_not_alerted() -> None:
    """One bad conversation is not a trend.

    A window with two turns once put an evaluator at 50% and would have paged
    on it.
    """
    df = scores([("s1", "flight_direction", 0.0), ("s2", "flight_direction", 0.0)])
    report = thresholds.evaluate(df, config())
    assert report.skipped == {"flight_direction": 2}
    assert not report.breaches
    assert "flight_direction" not in report.measured


def test_paging_is_narrower_than_alerting() -> None:
    df = scores(
        [(f"a{i}", "flight_direction", 0.0) for i in range(4)]
        + [(f"b{i}", "hallucination", 0.0) for i in range(4)]
    )
    report = thresholds.evaluate(df, config())
    assert [b.evaluator for b in report.breaches] == ["flight_direction"]
    assert not report.should_page, "flight_direction alerts but is not on page_on"


def test_empty_window_is_not_an_error() -> None:
    """A quiet hour is normal. Monitoring that raises then looks like an incident."""
    report = thresholds.evaluate(pd.DataFrame(), config())
    assert report.breaches == [] and report.measured == {}


# --------------------------------------------------------------------------
# curate
# --------------------------------------------------------------------------

def test_failure_detection_is_direction_aware() -> None:
    df = scores([("s1", "hallucination", 1.0), ("s2", "flight_direction", 1.0)])
    failing = curate._failing(df, "travel")
    assert failing == {"s1": ["hallucination"]}, "1.0 is a failure only for minimize"


def test_failure_detection_uses_the_agent_it_is_given() -> None:
    """Regression test.

    `_failing` once had `for_agent("travel")` hardcoded. For any other agent no
    evaluators resolved, every direction fell back to "maximize", and a
    minimize metric was scored exactly backwards — failures counted as passes.
    """
    df = scores([("s1", "hallucination", 1.0)])
    assert curate._failing(df, "travel") == {"s1": ["hallucination"]}
    assert curate._failing(df, "nonexistent-agent") == {}, (
        "an unknown agent must resolve no evaluators, not silently default"
    )


def test_expected_values_are_derived_where_the_fixtures_can_say() -> None:
    call = {"name": "search_flights",
            "input": {"origin": "New York", "destination": "Miami", "date": "2026-10-01"},
            "output": []}
    expected, needs_review, _ = curate.derive_expected(turn("s1", "q", [call]), T)
    assert not needs_review
    assert expected["expected_result_ids"] == sorted(
        f["flight_number"] for f in T.expected_flights("New York", "Miami", "2026-10-01")
    )


def test_reply_level_failures_are_flagged_for_review_not_guessed() -> None:
    """The tool call was right and the prose was wrong: nothing is computable."""
    call = {"name": "get_weather", "input": {"city": "Miami", "date": "2026-10-01"}, "output": {}}
    _, needs_review, reason = curate.derive_expected(turn("s1", "q", [call]), T)
    assert needs_review and "human" in reason

    _, needs_review, _ = curate.derive_expected(turn("s2", "q", []), T)
    assert needs_review, "no tool call means no derivable expectation"


def test_curation_respects_its_per_run_cap() -> None:
    rows = [(f"s{i}", "flight_direction", 0.0) for i in range(10)]
    records = [turn(f"s{i}", "q") for i in range(10)]
    got = curate.collect(records, scores(rows), config(curation={"max_per_run": 3}), T)
    assert len(got) == 3


# --------------------------------------------------------------------------
# diagnose
# --------------------------------------------------------------------------

def test_clusters_only_contain_failures_and_respect_direction() -> None:
    df = scores([("s1", "hallucination", 1.0), ("s2", "hallucination", 0.0)])
    records = [turn("s1", "q1"), turn("s2", "q2")]
    clusters = diagnose.build(records, df, config())
    assert len(clusters) == 1
    assert clusters[0].failed == 1 and clusters[0].applicable == 2
    assert [t["span_id"] for t in clusters[0].turns] == ["s1"]


def test_shared_features_require_every_failing_turn() -> None:
    """A feature true of only some failures is not the discriminator."""
    records = [
        turn("s1", "a flight next Friday", []),
        turn("s2", "a flight on 2026-10-01", []),
    ]
    df = scores([("s1", "graceful_alternative", 0.0), ("s2", "graceful_alternative", 0.0)])
    shared = diagnose.build(records, df, config())[0].shared_features()
    assert "no tool call was made" in shared
    assert "the request contains a relative date" not in shared, "true of s1 only"


def test_shared_features_surface_a_common_relative_date() -> None:
    records = [turn("s1", "a flight next Friday", []), turn("s2", "a hotel this weekend", [])]
    df = scores([("s1", "graceful_alternative", 0.0), ("s2", "graceful_alternative", 0.0)])
    shared = diagnose.build(records, df, config())[0].shared_features()
    assert "the request contains a relative date" in shared


def test_breached_only_would_have_hidden_the_finding_that_mattered() -> None:
    """The relative-date failures sat under the sample floor and never breached.

    Diagnosis therefore defaults to every failing turn, not just breaches.
    """
    records = [turn(f"s{i}", "q", []) for i in range(2)]
    df = scores([(f"s{i}", "graceful_alternative", 0.0) for i in range(2)])
    conf = config(thresholds={"graceful_alternative": {"min_rate": 0.8}})
    assert diagnose.build(records, df, conf) != []
    assert diagnose.build(records, df, conf, breached_only=True) != []

    passing = scores([(f"s{i}", "graceful_alternative", 1.0) for i in range(2)])
    assert diagnose.build(records, passing, conf) == []


def test_evidence_renders_the_evaluator_explanations_verbatim() -> None:
    """Co-locating the explanations is what actually solves these cases."""
    records = [turn("s1", "a flight next Friday", [], reply="what date?")]
    df = scores([("s1", "graceful_alternative", 0.0)])
    text = diagnose.evidence(diagnose.build(records, df, config()))
    assert "graceful_alternative" in text
    assert "because graceful_alternative said so" in text
    assert "what date?" in text


def test_diagnose_handles_an_empty_window() -> None:
    assert diagnose.build([], pd.DataFrame(), config()) == []
    assert "No failing turns" in diagnose.evidence([])


# --------------------------------------------------------------------------
# regression tests for issues found in code review
# --------------------------------------------------------------------------

def test_cluster_rate_is_expressed_against_its_own_threshold() -> None:
    """`rate` once returned a pass rate for every evaluator.

    Compared against a `max_rate` ceiling that made any minimize evaluator with
    a single failure look permanently breached, and printed a digest header
    contradicting `thresholds.evaluate`.
    """
    rows = [(f"s{i}", "hallucination", 1.0 if i == 0 else 0.0) for i in range(100)]
    records = [turn(f"s{i}", "q") for i in range(100)]
    cluster = diagnose.build(records, scores(rows), config())[0]
    assert cluster.rate == pytest.approx(0.01), "minimize reports the FLAGGED rate"
    assert diagnose.build(records, scores(rows), config(), breached_only=True) == [], (
        "1% flagged is inside a 5% ceiling and must not breach"
    )

    ok = [(f"s{i}", "flight_direction", 0.0 if i == 0 else 1.0) for i in range(100)]
    maxi = diagnose.build(records, scores(ok), config())[0]
    assert maxi.rate == pytest.approx(0.99), "maximize still reports the PASS rate"


def test_a_malformed_date_is_never_labelled_as_no_availability() -> None:
    """`expected_flights` compares ISO strings and never raises.

    A malformed date therefore matches nothing, and deriving from that would
    publish "there are no flights on this route" as ground truth when the real
    defect was the date the agent sent.
    """
    call = {"name": "search_flights",
            "input": {"origin": "New York", "destination": "Miami", "date": "March 12, 2026"},
            "output": []}
    expected, needs_review, reason = curate.derive_expected(turn("s1", "q", [call]), T)
    assert needs_review and "malformed" in reason
    assert expected == {}, "nothing may be asserted about availability here"


def test_human_annotations_are_not_read_as_evaluator_failures() -> None:
    """Span annotations include `user_feedback` and anything the UI wrote.

    An unknown name once fell through to direction="maximize", turning a human
    thumbs-down into a curation candidate attributed to an evaluator.
    """
    df = scores([("s1", "user_feedback", 0.0), ("s2", "flight_direction", 0.0)])
    assert curate._failing(df, "travel") == {"s2": ["flight_direction"]}


def test_an_evaluator_that_produced_no_scores_is_surfaced() -> None:
    """A judge outage must not read as a clean bill of health.

    `monitor.score` records a failed judge as score=None, so a rate limit or a
    bad key across the whole window leaves rows with nothing scored.
    """
    df = pd.DataFrame([
        {"span_id": f"s{i}", "annotation_name": "hallucination",
         "score": None, "label": "error", "explanation": "RateLimitError"}
        for i in range(5)
    ])
    report = thresholds.evaluate(df, config())
    assert report.unscored == ["hallucination"]
    assert report.is_degraded
    assert "NO USABLE SCORES" in report.summary()


def test_unknown_config_keys_fail_loudly(tmp_path) -> None:
    """`threshold:` for `thresholds:` once produced an agent that never alerts."""
    path = tmp_path / "config.yaml"
    path.write_text(
        "agent: x\nproject: p\ndataset: d\nagent_span_name: s\n"
        "threshold:\n  flight_direction: {min_rate: 0.99}\n"
    )
    with pytest.raises(ValueError, match="unknown config key"):
        agent_config.load(path)


def test_tool_span_attributes_survive_a_non_recording_span() -> None:
    """Regression test for a CI-only failure.

    With a provider installed but nothing exporting — a pytest run with tracing
    switched off — the tracer is a real OITracer that accepts
    `openinference_span_kind`, but hands back an OpenInferenceSpan wrapping a
    NonRecordingSpan. That object *has* `set_input`; calling it raises "Cannot
    set input attributes on a non-OpenInference span". Guarding the tracer was
    not enough; the span has to be checked too.

    Tracing is a side effect and must never take down the turn it observes.
    """
    from agent.loop import _record, _record_output, _tool_span

    class NonRecording:
        def is_recording(self) -> bool:
            return False

        def set_input(self, value, **kw):
            raise AssertionError("must not be called on a non-recording span")

    _record(NonRecording(), "set_input", {"a": 1})  # must not raise

    class RecordingButNotOpenInference:
        def is_recording(self) -> bool:
            return True

        def set_input(self, value, **kw):
            raise ValueError("Cannot set input attributes on a non-OpenInference span")

    _record(RecordingButNotOpenInference(), "set_input", {"a": 1})  # must not raise
    _record_output(NonRecording(), [{"flight_number": "DL 883"}])  # must not raise

    # And end to end with whatever tracer this process actually has.
    with _tool_span("search_flights", {"origin": "New York"}) as span:
        _record_output(span, [{"flight_number": "DL 883"}])


# --------------------------------------------------------------------------
# evaluators added from axial coding over production traces
# --------------------------------------------------------------------------

def test_a_past_date_is_still_searched() -> None:
    """Trace coding found the agent refusing one past date and serving another
    in the same run — 'March 12, 2026' declined as past while 'April 20, 2026'
    was searched. The policy now says search either way; this pins it."""
    from evals.agents.travel.evaluators import past_date_still_searched as ev

    refused = {"input": {"message": "Find me a flight from New York to Miami on March 12, 2026."},
               "output": {"reply": "That date is in the past.", "tool_calls": []}}
    assert ev.evaluate(refused)[0].score == 0.0

    # A compliant call carries the requested date; an empty input would mean the
    # date never reached the tool, which is the failure, not the pass.
    searched = {"input": {"message": "Find me a flight from New York to Miami on March 12, 2026."},
                "output": {"reply": "That date has passed, but here is what ran.",
                           "tool_calls": [{"name": "search_flights",
                                           "input": {"origin": "New York",
                                                     "destination": "Miami",
                                                     "date": "2026-03-12"},
                                           "output": []}]}}
    assert ev.evaluate(searched)[0].score == 1.0

    future = {"input": {"message": "A flight on December 30, 2027."},
              "output": {"reply": "Let me check.", "tool_calls": []}}
    assert ev.evaluate(future)[0].score == 1.0, "a future date imposes no obligation here"

    # Compliance is the requested date reaching a tool, not merely that some
    # tool ran: get_weather also takes a date, so a tool-name check alone scored
    # a refusal-plus-unrelated-call as compliant.
    sidestep = {"input": {"message": "A flight on March 12, 2026."},
                "output": {"reply": "That date is past. Here is the weather.",
                           "tool_calls": [{"name": "get_weather",
                                           "input": {"city": "Miami", "date": "2026-09-05"},
                                           "output": {}}]}}
    assert ev.evaluate(sidestep)[0].score == 0.0

    substituted = {"input": {"message": "A flight on March 12, 2026."},
                   "output": {"reply": "Searching 2027 instead.",
                              "tool_calls": [{"name": "search_flights",
                                              "input": {"date": "2027-03-12"}, "output": []}]}}
    assert ev.evaluate(substituted)[0].score == 0.0, "silently searching a different date is a refusal"

    # Not every tool accepts a date. create_itinerary takes destination and
    # num_days, so a past date can never reach an argument even when the agent
    # fulfils the request correctly — this was a live false positive on real
    # traffic before the check learned the difference.
    undated_tool = {"input": {"message": "A 5-day itinerary for Paris, arriving June 10, 2026."},
                    "output": {"reply": "Here is your itinerary.",
                               "tool_calls": [{"name": "create_itinerary",
                                               "input": {"destination": "Paris", "num_days": 5},
                                               "output": {}}]}}
    # Scored None, not 1.0. The request was plausibly honoured, but nothing in
    # the turn proves it, and counting an unverifiable turn as a pass inflates
    # the metric that gates on it. `sweep()` drops null scores, so this leaves
    # the denominator rather than padding it.
    result = ev.evaluate(undated_tool)[0]
    assert result.score is None
    assert result.label == "unverifiable"


def test_unverifiable_turns_leave_the_denominator() -> None:
    """A null score must not be logged as a pass or a failure.

    This is the property the None relies on: `sweep` drops null-scored rows
    before writing annotations, so an unverifiable turn is absent from the rate
    rather than counted in it.
    """
    scored = pd.DataFrame([
        {"span_id": "a", "annotation_name": "past_date_still_searched", "score": 1.0},
        {"span_id": "b", "annotation_name": "past_date_still_searched", "score": None},
        {"span_id": "c", "annotation_name": "past_date_still_searched", "score": 0.0},
    ])
    kept = scored.dropna(subset=["score"])
    assert len(kept) == 2
    assert kept["score"].mean() == 0.5, "the null turn must not shift the rate"


def test_a_stated_total_must_match_the_arithmetic() -> None:
    """Derived values were the largest taxonomy category. Sound arithmetic is
    fine; wrong arithmetic reads identically to a customer."""
    from evals.agents.travel.evaluators import derived_totals_are_correct as ev

    call = {"name": "search_hotels",
            "input": {"city": "Paris", "check_in": "2026-06-10", "check_out": "2026-06-14"},
            "output": [{"name": "Lumiere", "price_per_night_usd": 385, "rating": 4.7}]}

    def score(reply: str) -> float:
        return ev.evaluate({"output": {"reply": reply, "tool_calls": [call]}})[0].score

    assert score("Lumiere $385/night — $1,540 for 4 nights.") == 1.0
    assert score("Lumiere $385/night — $1,155 for 4 nights.") == 0.0, "wrong total"
    assert score("Lumiere is $410/night.") == 0.0, "rate absent from the tool output"
    assert score("Lumiere $385/night, rated 4.7 out of 5.") == 1.0, "a rating is not money"


def test_totals_pair_rates_with_their_own_stay_length() -> None:
    """Rates and nights were pooled across calls, which failed correct replies
    two ways: a total from an earlier call's date range looked fabricated, and
    one malformed call reset the night count, dropping every valid total."""
    from evals.agents.travel.evaluators import derived_totals_are_correct as ev

    def hotels(ci, co, rate):
        return {"name": "search_hotels",
                "input": {"city": "X", "check_in": ci, "check_out": co},
                "output": [{"name": "H", "price_per_night_usd": rate}]}

    def score(reply, calls):
        return ev.evaluate({"output": {"reply": reply, "tool_calls": calls}})[0].score

    two = [hotels("2026-06-10", "2026-06-14", 385), hotels("2026-08-07", "2026-08-09", 300)]
    assert score("$385/night — $1,540 for 4 nights. $300/night — $600 for 2 nights.", two) == 1.0

    malformed = [hotels("2026-06-10", "2026-06-14", 385),
                 {"name": "search_hotels", "input": {"city": "X"}, "output": []}]
    assert score("$385/night — $1,540 for 4 nights.", malformed) == 1.0

    assert score("$385/night — $1,155 for 4 nights.", [hotels("2026-06-10", "2026-06-14", 385)]) == 0.0


# --------------------------------------------------------------------------
# session identity — what the judges' applicability gate depends on
# --------------------------------------------------------------------------

def test_every_tool_call_reaches_the_judge_not_just_the_first() -> None:
    """A multi-tool turn must show the judge all of its evidence.

    Passing only call #1 made facts sourced from calls #2..n look invented:
    four `get_weather` calls, one per night, read as three fabricated days.
    """
    calls = [
        {"name": "get_weather", "input": {"date": "2026-05-01"},
         "output": {"condition": "Sunny"}},
        {"name": "get_weather", "input": {"date": "2026-05-02"},
         "output": {"condition": "Humid"}},
    ]
    rendered = judges._all_calls(turn("s1", "weather?", calls), "output")
    assert "Sunny" in rendered
    assert "Humid" in rendered, "evidence from the second call was dropped"
    assert "get_weather" in rendered, "the judge cannot tell which result is which"


def test_a_single_call_is_still_rendered_unwrapped() -> None:
    """The common case keeps its original shape — the judge was validated on it."""
    calls = [{"name": "search_flights", "input": {}, "output": {"airline": "Delta"}}]
    assert judges._all_calls(turn("s1", "flights?", calls), "output") == '{"airline": "Delta"}'


def test_graceful_alternative_skips_a_follow_up_answered_from_context() -> None:
    """"Which of those is cheapest?" calls no tool, but nothing was unavailable.

    Grading it asks the judge how well the agent handled having nothing to
    offer, when it had plenty — and the judge duly says so, which lands as a
    failure. This is what dragged the metric from 84.6% to 73.8%.
    """
    assert not judges._no_results(
        turn("s1", "Which of those is cheapest?", calls=[], turn_index=2)
    )


def test_graceful_alternative_still_covers_an_opening_out_of_scope_request() -> None:
    """The case the evaluator exists for must survive the tightening."""
    assert judges._no_results(
        turn("s1", "Do I need a visa to visit Japan?", calls=[], turn_index=1)
    )


def test_graceful_alternative_covers_empty_results_at_any_turn() -> None:
    """A search that found nothing is a no-results turn wherever it happens."""
    empty = [{"name": "search_flights", "input": {}, "output": []}]
    assert judges._no_results(turn("s1", "flights to Lagos?", empty, turn_index=4))


def test_spans_without_a_turn_index_keep_the_old_behaviour() -> None:
    """Spans predating session identity must not vanish from the denominator."""
    assert judges._no_results(turn("s1", "anything?", calls=[], turn_index=None))


def test_turn_index_is_read_from_the_nested_attribute_shape() -> None:
    """Phoenix nests dotted attribute namespaces; session.id stays flat.

    Reading only the flat column returned None for every span, which silently
    disabled the follow-up gate rather than failing — the exact shape of bug
    that reaches production looking like "the fix didn't work".
    """
    spans = pd.DataFrame([
        {"name": "travel_agent", "context.span_id": "s1", "context.trace_id": "t1",
         "parent_id": None, "span_kind": "AGENT",
         "attributes.input.value": "Which of those is cheapest?",
         "attributes.output.value": "The Delta one, $214.",
         "attributes.session.id": "conv-1",
         "attributes.travel_agent": {"turn_index": 2}},
    ])
    rec = monitor.reconstruct(spans)[0]
    assert rec["session_id"] == "conv-1"
    assert rec["input"]["turn_index"] == 2
    assert not judges._no_results(rec)


# --------------------------------------------------------------------------
# capability disclosure
# --------------------------------------------------------------------------

def _booking(reply: str, message: str = "Hold the New York to Chicago flight "
                                        "on October 8, 2026 — put the deposit on my card.") -> float:
    from evals.agents.travel.evaluators import booking_limits_disclosed
    return booking_limits_disclosed.evaluate(
        {"input": {"message": message}, "output": {"reply": reply, "tool_calls": []}}
    )[0].score


def test_listing_flights_without_saying_it_cannot_book_fails() -> None:
    """The production failure: search results, limitation left unsaid."""
    assert _booking(
        "Here are your options on October 8, 2026: United UA 512 departs 6:50am — $178."
    ) == 0.0


def test_a_plain_disclosure_passes() -> None:
    assert _booking(
        "I can't hold a flight or take a deposit — I can only search. "
        "Here's what's available on October 8: United UA 512 at $178."
    ) == 1.0


def test_implying_the_booking_happened_fails_even_with_a_disclosure() -> None:
    """Worse than silence: the user acts on a seat that was never held."""
    assert _booking(
        "I can't process payments directly, but you're all set on United UA 512."
    ) == 0.0


def test_non_transactional_requests_are_not_graded() -> None:
    assert _booking("Here are the flights I found.",
                    message="What flights are there from Denver to Chicago?") == 1.0


def test_disclosure_is_not_matched_by_the_prompt_s_own_wording() -> None:
    """Guard against a tautological evaluator.

    If this only recognised the phrasing v5 asks for, it would prove the prompt
    was pasted rather than that the agent disclosed anything. Wordings the
    prompt never mentions must still pass.
    """
    for reply in (
        "Unfortunately I'm unable to book that for you, but here's what's free.",
        "I have no way to reserve seats — you'd need to book directly.",
        "I am not able to complete a purchase, though I can show you the options.",
    ):
        assert _booking(reply) == 1.0, reply


def test_real_disclosure_wordings_from_production_are_recognised() -> None:
    """Pinned from the v4 baseline, where both scored as silence.

    `\\bbook\\b` misses "bookings" and `\\breserve\\b` misses "reservations".
    A paging invariant that fires on correct behaviour gets muted, so its
    false-negative rate matters as much as its false-positive rate.
    """
    assert _booking(
        "I can't help with changes to existing reservations. What I can help "
        "with is booking flights, finding hotels, and checking weather.",
        message="Can you change the name on my existing reservation?",
    ) == 1.0
    assert _booking(
        "I can't help with refunds or manage bookings you've already made — "
        "that's something you'd need to take up with the airline.",
        message="I booked a flight through you last month and need a refund.",
    ) == 1.0


# --------------------------------------------------------------------------
# cost reporting
# --------------------------------------------------------------------------

def _cost_report(**over):
    from evals.core import cost
    base = {"agent": "travel", "window_minutes": 180, "conversations": 100,
            "turns": 200, "agent_cost": 1.0, "agent_tokens": 1_000_000.0,
            "eval_cost": 2.0, "eval_tokens": 500_000.0}
    base.update(over)
    return cost.CostReport(**base)


def test_cost_is_expressed_per_conversation() -> None:
    """The unit the customer asked about. Phoenix reports totals."""
    r = _cost_report(agent_cost=5.0, conversations=100, turns=250)
    assert r.cost_per_conversation == 0.05
    assert r.cost_per_turn == 0.02


def test_eval_overhead_is_reported_even_when_it_exceeds_the_agent() -> None:
    """Adopting this system is itself an operating cost.

    A cost view that reports only the agent hides the half of the bill the
    customer takes on by running the evaluation stack.
    """
    r = _cost_report(agent_cost=1.0, eval_cost=2.44)
    assert r.eval_overhead == 2.44
    assert "evals cost more than the agent" in r.summary()


def test_an_empty_window_reports_nothing_rather_than_dividing_by_zero() -> None:
    r = _cost_report(conversations=0, turns=0, agent_cost=0.0, agent_tokens=0.0)
    assert r.cost_per_conversation is None
    assert "no traffic in the window" in r.summary()
    assert r.breaches == []


def test_cost_breaches_only_against_a_configured_ceiling() -> None:
    """No unit economic was given, so absence of a ceiling must not alert."""
    assert _cost_report(agent_cost=100.0).breaches == []
    r = _cost_report(agent_cost=100.0, conversations=100, max_cost_per_conversation=0.05)
    assert len(r.breaches) == 1 and "above the $0.0500 ceiling" in r.breaches[0]


def test_projection_uses_the_stated_annual_volume() -> None:
    r = _cost_report(agent_cost=1.0, conversations=100, annual_conversations=1_000_000)
    assert r.projected_annual_cost == 10_000.0


def test_a_project_with_no_traces_is_named_not_silently_zeroed() -> None:
    """Under-reporting the bill is the one failure a cost view must not have."""
    r = _cost_report(missing_projects=["evaluators"])
    assert "'evaluators' has no traces" in r.summary()
