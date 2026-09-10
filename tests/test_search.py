"""Authority search: the engine run in the other direction.

A lawyer with a proposition and no citation asks which judgment backs it and where. These tests pin
the two things that separate that from a text search: a passage that is not the court speaking is not
an authority however well its words match, and the answer names a line rather than a document.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import text as sql_text

from orderorder.db.models import CitationAlias, Judgment
from orderorder.engine.search import (
    MAX_NEAR_QUERIES,
    NEAR_TERMS,
    ROLE_PRIOR,
    best_line,
    build_index,
    find_authorities,
    fts_query,
    index_exists,
    near_queries,
    search_paragraphs,
)
from orderorder.ingest.pdf import ExtractedJudgment
from orderorder.ingest.store import store_extracted

HOLDING_JUDGMENT = """1. Leave granted.

2. It was strenuously contended on behalf of the appellant that any misrepresentation whatsoever
vitiates consent in a commercial contract, however immaterial the misstatement.

3. A misrepresentation vitiates consent only where it induced the contract. The burden of proving
inducement lies upon the party alleging it. Mere inaccuracy in a recital is not enough.

4. In view of the above, the appeals are dismissed with no order as to costs.
"""

UNRELATED_JUDGMENT = """1. Leave granted.

2. The question is whether the tenant was entitled to notice under Section 106 of the Transfer of
Property Act before the suit for eviction was instituted.

3. A notice under Section 106 is mandatory and its absence is fatal to a suit for eviction.

4. The appeal is allowed.
"""

BENCH_JUDGMENT = """1. Leave granted.

2. A misrepresentation vitiates consent only where it induced the contract, and this Court has said
so consistently since the Contract Act was enacted.

3. The appeal is dismissed.
"""


def _add(session, key: str, title: str, text: str, *, bench: int, year: int, headnote: str = "") -> Judgment:
    judgment = Judgment(
        canonical_key=key,
        court="Supreme Court of India",
        title=title,
        source="aws_open_data",
        source_id=key,
        bench_strength=bench,
        decided_on=dt.date(year, 6, 1),
    )
    session.add(judgment)
    session.flush()
    session.add(
        CitationAlias(
            judgment_id=judgment.id,
            reporter="SCC",
            citation_string=f"({year}) 1 SCC {bench}",
            normalized=f"SCC:{year}:1:{bench}",
        )
    )
    store_extracted(
        session,
        judgment,
        ExtractedJudgment(source_path=key, page_count=4, headnote=headnote, judgment=text),
    )
    return judgment


@pytest.fixture
def corpus(session):
    _add(session, "INSC:2019:1", "ALPHA versus BETA", HOLDING_JUDGMENT, bench=2, year=2019)
    _add(session, "INSC:2020:2", "GAMMA versus DELTA", UNRELATED_JUDGMENT, bench=2, year=2020)
    session.commit()
    build_index(session)
    return session


CLAIM = "a misrepresentation vitiates consent only where it induced the contract"


def test_the_index_is_built_over_stored_paragraphs(corpus) -> None:
    assert index_exists(corpus)
    rows = search_paragraphs(corpus, CLAIM)
    assert rows


def test_the_holding_is_found_across_the_corpus(corpus) -> None:
    found = find_authorities(corpus, CLAIM, top=3)
    assert found
    best = found[0]
    assert best.canonical_key == "INSC:2019:1"
    assert best.paragraph_label == "3"


def test_the_line_is_named_not_just_the_judgment(corpus) -> None:
    """The point of the exercise: which sentence of a long judgment to read."""
    best = find_authorities(corpus, CLAIM, top=1)[0]
    assert best.line == "A misrepresentation vitiates consent only where it induced the contract."
    assert best.line_start is not None
    assert best.body[best.line_start : best.line_end] == best.line


def test_counsels_submission_is_not_offered_as_authority(corpus) -> None:
    """Paragraph 2 states the rule more baldly than the court does, and matches the words better.

    An advocate's submission makes a better keyword match than a holding precisely because it is
    unqualified. Offering it as authority would hand a lawyer the mistake the verifier exists to catch.
    """
    found = find_authorities(corpus, "any misrepresentation whatsoever vitiates consent", top=5)
    assert all(a.paragraph_label != "2" for a in found)

    with_argument = find_authorities(
        corpus, "any misrepresentation whatsoever vitiates consent", top=5, court_voice_only=False
    )
    counsel = [a for a in with_argument if a.paragraph_label == "2"]
    assert counsel and counsel[0].voice is not None
    assert counsel[0].voice.voice == "counsel_argument"


def test_the_headnote_is_never_an_authority(session) -> None:
    """It is the publisher's summary of the judgment, not the court's words."""
    _add(
        session,
        "INSC:2021:3",
        "EPSILON versus ZETA",
        HOLDING_JUDGMENT,
        bench=2,
        year=2021,
        headnote="HELD: a misrepresentation vitiates consent only where it induced the contract.",
    )
    session.commit()
    build_index(session)
    found = find_authorities(session, CLAIM, top=5, one_per_judgment=False)
    assert found
    assert all(a.voice is None or a.voice.voice != "headnote" for a in found)
    assert all("HELD:" not in a.body for a in found)


def test_a_larger_bench_outranks_a_smaller_one_on_the_same_point(corpus) -> None:
    _add(corpus, "INSC:2018:9", "CONSTITUTION BENCH", BENCH_JUDGMENT, bench=7, year=2018)
    corpus.commit()
    build_index(corpus, rebuild=True)

    found = find_authorities(corpus, CLAIM, top=3)
    assert found[0].canonical_key == "INSC:2018:9"
    assert found[0].bench_strength == 7


def test_one_result_per_judgment_by_default(corpus) -> None:
    found = find_authorities(corpus, CLAIM, top=5)
    keys = [a.canonical_key for a in found]
    assert len(keys) == len(set(keys))


def test_a_proposition_nothing_matches_returns_nothing(corpus) -> None:
    assert find_authorities(corpus, "maritime salvage of a derelict vessel", top=3) == []


def test_a_query_of_only_stopwords_is_not_a_search(corpus) -> None:
    assert fts_query("of the and to") == ""
    assert search_paragraphs(corpus, "of the and to") == []


def test_the_authority_names_its_pinpoint(corpus) -> None:
    best = find_authorities(corpus, CLAIM, top=1)[0]
    assert best.pinpoint == "(2019) 1 SCC 2, para 3"


# --- picking the line ---------------------------------------------------------


def test_the_densest_sentence_wins_not_the_longest() -> None:
    body = (
        "The appellant relied upon a number of decisions of this Court and of the High Courts, none of "
        "which was shown to bear upon the question of inducement now raised before us in this appeal. "
        "A misrepresentation vitiates consent only where it induced the contract."
    )
    line, start, end = best_line(body, "misrepresentation vitiates consent where it induced the contract")
    assert line == "A misrepresentation vitiates consent only where it induced the contract."
    assert body[start:end] == line


def test_a_paragraph_sharing_nothing_yields_no_line() -> None:
    assert best_line("The appeal is dismissed.", "maritime salvage") == (None, None, None)


# --- words together, not merely words --------------------------------------------------------------


def test_a_proposition_raises_proximity_queries_over_runs_of_its_terms() -> None:
    proposition = (
        "A misrepresentation vitiates consent only where it induced the contract and the burden lies"
    )
    near = near_queries(proposition)
    assert " OR " in fts_query(proposition)
    assert near, "a proposition of this length should raise at least one proximity query"
    for query in near:
        assert query.startswith("NEAR(")
        assert query.count('"') == NEAR_TERMS * 2


def test_a_proposition_too_short_to_have_a_run_raises_none() -> None:
    assert near_queries("consent vitiated") == []


def test_proximity_queries_are_capped_however_long_the_proposition() -> None:
    """Each one is a query against the corpus; a page of text must not become a page of queries."""
    long_one = " ".join(f"distinctive{n} holding{n} principle{n}" for n in range(40))
    assert len(near_queries(long_one)) <= MAX_NEAR_QUERIES


def test_the_paragraph_the_words_came_from_beats_one_that_merely_shares_them(corpus) -> None:
    """The point of proximity: a long paragraph on the same subject carries more of the vocabulary.

    Here BENCH_JUDGMENT's paragraph 2 says the rule and goes on at length; HOLDING_JUDGMENT's
    paragraph 3 is where the sentence actually is. Asking which paragraph has the words *together*
    is what separates them.
    """
    lifted = (
        "A misrepresentation vitiates consent only where it induced the contract. The burden of "
        "proving inducement lies upon the party alleging it."
    )
    found = find_authorities(corpus, lifted, top=3, one_per_judgment=False)
    assert found
    assert found[0].canonical_key == "INSC:2019:1"
    assert found[0].paragraph_label == "3"


def test_relevance_reported_is_the_one_the_ranking_used(corpus) -> None:
    """If the rows carried the BM25 score, the caller would re-sort by the ranker fusion corrects."""
    found = find_authorities(corpus, CLAIM, top=3, one_per_judgment=False)
    assert found
    assert [a.score for a in found] == sorted((a.score for a in found), reverse=True)
    assert all(a.relevance > 0 for a in found)


# --- keeping the index level with the paragraphs -----------------------------------------------------

SECOND_HOLDING = """1. Leave granted.

2. A notice under Section 106 of the Transfer of Property Act is mandatory before a suit for
eviction, and its absence is fatal to the suit however the tenant may have behaved.

3. The appeal is allowed.
"""


def test_a_judgment_ingested_later_is_indexed_by_the_next_run(corpus) -> None:
    """The batched workflow: ingest some, index, ingest more, index again.

    `index` used to return the moment the table held any rows, so the second batch was never indexed
    and `--rebuild` -- re-tokenising the whole corpus -- was the only way to see it. That is the cost
    of the entire ingestion again, on the corpus that most needs ingesting in batches.
    """
    before = build_index(corpus)
    _add(corpus, "INSC:2021:3", "EPSILON versus ZETA", SECOND_HOLDING, bench=2, year=2021)
    corpus.commit()

    after = build_index(corpus)
    assert after > before
    found = find_authorities(corpus, "a notice under Section 106 is mandatory before a suit for eviction")
    assert any(a.canonical_key == "INSC:2021:3" for a in found)


def test_paragraphs_replaced_by_a_re_ingest_do_not_linger_in_the_index(corpus) -> None:
    """`store_extracted` deletes a version's paragraphs and writes new ones with new ids.

    `_match` reads the body straight out of the index, so a row left behind serves superseded text
    under a paragraph id that no longer exists.
    """
    judgment = corpus.query(Judgment).filter_by(canonical_key="INSC:2019:1").one()
    store_extracted(
        corpus,
        judgment,
        ExtractedJudgment(
            source_path="INSC:2019:1", page_count=4, headnote="", judgment=SECOND_HOLDING
        ),
    )
    corpus.commit()
    build_index(corpus)

    live = {row[0] for row in corpus.execute(sql_text("SELECT id FROM paragraph")).all()}
    indexed = {
        row[0] for row in corpus.execute(sql_text("SELECT paragraph_id FROM paragraph_fts")).all()
    }
    assert indexed <= live, f"{len(indexed - live)} indexed paragraphs no longer exist"
    assert not find_authorities(corpus, CLAIM), "the replaced text is still being served"


def test_running_the_index_twice_changes_nothing(corpus) -> None:
    """Reconciling has to be idempotent, or every run would grow the index by a whole corpus."""
    first = build_index(corpus)
    assert build_index(corpus) == first


def test_the_role_prior_moves_a_holding_above_a_matching_narration(corpus) -> None:
    """Same words, different roles: the paragraph that states the law outranks one that recites it."""
    rows = search_paragraphs(corpus, CLAIM)
    assert rows, "the corpus must answer the claim at all"

    # Every paragraph starts unlabelled, so all carry the narration prior; labelling one `ratio`
    # lifts its relevance by exactly the prior difference and nothing else moves.
    before = {
        r["paragraph_id"]: r["relevance"] for r in search_paragraphs(corpus, CLAIM, role_boost=True)
    }
    target = rows[0]["paragraph_id"]
    corpus.execute(
        sql_text("UPDATE paragraph SET role = 'ratio' WHERE id = :pid"), {"pid": target}
    )
    corpus.commit()
    after = {
        r["paragraph_id"]: r["relevance"] for r in search_paragraphs(corpus, CLAIM, role_boost=True)
    }

    # Unlabelled rows carry a zero prior; labelling to ratio adds the holding prior in full.
    delta = ROLE_PRIOR["ratio"]
    assert after[target] == pytest.approx(before[target] + delta, abs=1e-6)


def test_the_role_filter_returns_only_the_roles_asked_for(corpus) -> None:
    rows = search_paragraphs(corpus, CLAIM, roles=["ratio"])
    assert all(r["role"] == "ratio" for r in rows)

    # And asking for a role nothing carries returns nothing rather than everything.
    assert search_paragraphs(corpus, CLAIM, roles=["disposition"]) == []


def test_find_authorities_honours_the_role_filter(corpus) -> None:
    found = find_authorities(corpus, CLAIM, top=5, roles=["argument_petitioner"])
    assert found == []
