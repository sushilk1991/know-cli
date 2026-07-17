"""Adversarial regressions for natural-language query classification."""

import pytest

from know.file_categories import apply_category_demotion
from know.query import analyze_query


def test_normal_capitalized_phrase_is_not_misclassified_as_camel_case():
    plan = analyze_query("Authentication flow")

    assert plan.identifiers == []
    assert plan.terms == ["Authentication", "flow"]
    assert plan.query_type == "concept"


def test_two_capital_camel_case_identifier_is_still_detected():
    assert analyze_query("AThing").identifiers == ["AThing"]


@pytest.mark.parametrize("query", ["tissue parser", "issue_tracker design"])
def test_error_words_require_whole_token_matches(query):
    assert analyze_query(query).query_type != "error"


@pytest.mark.parametrize(
    "query",
    [
        "errors in parser",
        "exceptions thrown",
        "failures in auth",
        "bugs in auth",
        "crashes at startup",
        "tests are failing",
    ],
)
def test_plural_and_inflected_error_words_keep_error_intent(query):
    assert analyze_query(query).query_type == "error"


@pytest.mark.parametrize("query", ["contest parser", "latest parser", "attestation parser"])
def test_test_demotion_override_requires_standalone_test_word(query):
    chunks = [{"file_path": "tests/test_parser.py", "score": 1.0}]

    apply_category_demotion(chunks, query=query)

    assert chunks[0]["score"] == pytest.approx(0.3)


@pytest.mark.parametrize("query", ["test parser", "tests parser"])
def test_explicit_test_query_still_disables_test_demotion(query):
    chunks = [{"file_path": "tests/test_parser.py", "score": 1.0}]

    apply_category_demotion(chunks, query=query)

    assert chunks[0]["score"] == 1.0


def test_unicode_identifier_keeps_its_leading_character():
    plan = analyze_query("Δelta_value")

    assert plan.identifiers == ["Δelta_value"]
    assert "elta_value" not in plan.all_search_terms
