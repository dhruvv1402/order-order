"""The engine's checks, as tools an agent may call.

Every function here is a wrapper. It opens a session, calls one entry point that already exists under
`orderorder.engine`, and returns the answer as plain JSON. No tool decides anything.

That division is the point, and it is the one docs/ARCHITECTURE.md already drew. `engine/graph.py`
says the verification graph is "deliberately not an agent: the model is called inside two nodes and
everything else is plain Python, so the path a citation takes is fixed and inspectable", and that
stays exactly true here. What the agent chooses is *which question to ask*; it never chooses the
answer. `verify_brief` below runs the same eight nodes in the same fixed order they have always run
in, and the agent cannot reorder them, skip one, or overrule a verdict. Every finding it reports is
one the engine produced and a reader can re-derive from the command line.

The docstrings are longer than the bodies because the docstring *is* the specification the model reads
when it chooses between these eight. Strands builds each tool's schema from the signature, the type
hints and these words, and a lawyer's question arrives as "is Kesavananda still good law" -- which has
to land on `check_treatment` and not on `find_authority`. Edit them as carefully as code.

Tools are built against a session factory rather than defined at module level, because the web
application injects its own (`create_app(session_factory=...)`) and the suite injects an in-memory
database. A tool holding `get_session` directly would read the developer's 1.2 GB corpus from inside
a test that believed it had an empty one, and pass.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy import func, select
from strands import tool

from orderorder.citations.grammar import extract_citations
from orderorder.db.models import Judgment, JudgmentTextVersion, Paragraph
from orderorder.db.session import get_session
from orderorder.engine import authority as authority_gate
from orderorder.engine import search
from orderorder.engine.citator import TreatmentReport, treatment_of
from orderorder.engine.contrary import ContraryReport, find_contrary
from orderorder.engine.graph import verify_text
from orderorder.engine.locator import LocationResult, locate
from orderorder.engine.providers import build_structured
from orderorder.engine.schemas import (
    ApplicabilityAssessment,
    ScopeAssessment,
    VoiceAssessment,
    WeightAssessment,
)
from orderorder.engine.search import Authority
from orderorder.ingest.store import load_paragraphs
from orderorder.resolver import resolve

# How much of a paragraph a tool hands back. A judgment's paragraph runs to a few thousand characters
# and five of them would be most of a context window, spent on text the agent is not being asked to
# read: it is being asked which paragraph, and the engine has already picked the line that decided
# that. The full text is one `orderorder judgment` away, and /api/judgment/{key} serves it to the page.
BODY_CHARS = 600

# Where a citation is checked without the corpus having the judgment's text, every downstream check is
# unanswerable and saying so is the answer. This is the sentence that says it, in one place.
NO_TEXT = "the corpus holds this judgment's metadata but not its text, so nothing in it can be read"


def _trim(body: str) -> str:
    if len(body) <= BODY_CHARS:
        return body
    return body[:BODY_CHARS].rstrip() + f" [...{len(body) - BODY_CHARS:,} more characters]"


def _judgment_from(session, key_or_citation: str) -> Judgment | None:
    """Find a judgment by canonical key, or failing that by reading the string as a citation.

    Both, because the agent will get this wrong in a predictable direction. The key is what every
    other tool returns and what the prompt asks for, but a user's question contains "(2019) 4 SCC 1"
    and a model that has just been handed that string will pass it here. Resolving it is a line of
    code and the alternative is a tool that answers "no judgment with key (2019) 4 SCC 1" to a
    question the corpus can perfectly well answer.
    """
    found = session.scalars(select(Judgment).where(Judgment.canonical_key == key_or_citation)).first()
    if found is not None:
        return found
    citations = extract_citations(key_or_citation)
    if not citations:
        return None
    resolution = resolve(session, citations[0])
    if not resolution.judgment_id:
        return None
    return session.get(Judgment, resolution.judgment_id)


def _authority_json(found: Authority) -> dict:
    """One paragraph offered as authority, as the agent needs it.

    `line` is the single sentence the search scored highest, and it is the one thing here the agent
    should quote. `doubtful` and `treatment_note` are carried on every authority rather than left for
    a follow-up call, because an agent that offers a paragraph from an overruled judgment and then
    checks its treatment afterwards has already given the wrong answer.
    """
    return {
        "key": found.canonical_key,
        "title": found.title,
        "citation": found.citation,
        "pinpoint": found.pinpoint,
        "paragraph": found.paragraph_label,
        "court": found.court,
        "decided_on": found.decided_on,
        "bench_strength": found.bench_strength,
        "score": round(found.score, 2),
        "line": found.line,
        "body": _trim(found.body),
        "voice": found.voice.voice if found.voice else None,
        "doubtful": bool(found.treatment and found.treatment.is_doubtful),
        "treatment_note": found.treatment.note if found.treatment else None,
    }


def _treatment_json(report: TreatmentReport) -> dict:
    """A treatment report, with the distinction that matters kept intact.

    `status` is "good_law" both for a judgment forty later benches have followed and for one nothing
    has ever cited, and only the first is a finding. `unchecked` separates them. Without it an agent
    reports "we checked and it is sound" about a judgment no court has ever mentioned, which is the
    single most damaging thing this tool could say.
    """
    return {
        "judgment": report.judgment_id,
        "status": report.status,
        "described": report.describe(),
        "note": report.note,
        "doubtful": report.is_doubtful,
        "unchecked": report.is_unchecked,
        "undermined": report.is_undermined,
        "citing_count": report.citing_count,
        "corpus_size": report.corpus_size,
        "negative": [
            {
                "citing": edge.citing_key,
                "title": edge.citing_title,
                "date": edge.citing_date,
                "bench": edge.citing_bench,
                "treatment": edge.treatment,
                "paragraph": edge.paragraph_label,
                "sentence": edge.sentence,
            }
            for edge in report.negative[:5]
        ],
    }


def _verdict_json(verdict: Any) -> dict:
    """One citation's verdict, shaped for a model that has to explain it in words.

    Deliberately not `web/jobs.as_json`, which is shaped for the page: that one carries spans and
    offsets so the browser can highlight a quote, and this one carries the findings' words and the
    reason a check was not run. An agent fed the page's payload spends its context on character
    offsets it cannot use and still has to be told why `support` is "not assessed".
    """
    scope = verdict.scope
    return {
        "citation": verdict.citation_raw,
        "proposition": verdict.proposition,
        "grade": verdict.grade,
        "existence": verdict.existence,
        "support": verdict.support,
        "case": verdict.judgment_title,
        "key": verdict.canonical_key,
        "paragraph": verdict.paragraph_label,
        "claimed_pinpoint": verdict.claimed_pinpoint,
        "quote_verified": verdict.quote_verified,
        "quote": verdict.quote if verdict.quote_verified else None,
        "needs_review": verdict.needs_review,
        "review_reason": verdict.review_reason,
        "narrowed_to": scope.narrowed_proposition if scope else None,
        "findings": [{"mode": f.mode, "label": f.label, "detail": f.detail} for f in verdict.findings],
        "treatment": None
        if verdict.treatment is None
        else {
            "status": verdict.treatment.status,
            "described": verdict.treatment.describe(),
            "doubtful": verdict.treatment.is_doubtful,
            "unchecked": verdict.treatment.is_unchecked,
        },
    }


def _contrary_json(report: ContraryReport) -> dict:
    return {
        "proposition": report.proposition,
        "found": report.found,
        "searched": report.searched,
        "described": report.describe(),
        "judgments_searched": report.judgments_searched,
        "paragraphs_examined": report.paragraphs_examined,
        "read_by_model": report.read_by_model,
        "leads": [
            {
                "key": lead.authority.canonical_key,
                "title": lead.authority.title,
                "pinpoint": lead.authority.pinpoint,
                "sentence": lead.sentence,
                "why": lead.opposition.describe(),
                "kind": lead.opposition.kind,
                "cue": lead.opposition.cue,
                # None means nobody read this pair, which is not the same as a model having read it
                # and found nothing. The engine keeps those apart and so does this.
                "confirmed": None if lead.reading is None else lead.reading.contradicts,
                "reading": None if lead.reading is None else lead.reading.describe(),
            }
            for lead in report.leads
        ],
    }


def _location_json(result: LocationResult) -> dict:
    return {
        "pinpoint": {
            "claimed": result.pinpoint.claimed,
            "status": result.pinpoint.status,
            "is_problem": result.pinpoint.is_problem,
            "exists": result.pinpoint.exists,
            "highest_label": result.pinpoint.highest_label,
            "paragraph_count": result.pinpoint.paragraph_count,
            "note": result.pinpoint.note,
            "found_at": result.pinpoint.found_at,
        },
        "query_terms": result.query_terms,
        "candidates": [
            {
                "paragraph": candidate.printed_label or candidate.seq,
                "seq": candidate.seq,
                "score": round(candidate.score, 2),
                "is_claimed_pinpoint": candidate.is_claimed_pinpoint,
                "likely_quoted": candidate.likely_quoted,
                "role": candidate.role,
                "opinion": candidate.opinion_kind,
                "matched_terms": candidate.matched_terms,
                "body": _trim(candidate.body),
            }
            for candidate in result.candidates
        ],
    }


def build_tools(session_factory: Callable[..., Any] = get_session) -> list:
    """Every check the agent may run, bound to one database.

    Returned as a list in the order a question usually travels: what is held, then the checks on a
    citation somebody wrote, then the searches for one nobody has written yet. Strands does not care
    about the order, but a reader of the system prompt does.
    """

    @tool
    def corpus_status() -> dict:
        """What this corpus actually holds: how many judgments, how many with text, the date range.

        Call this before answering any question about coverage ("do you have Kesavananda", "what can
        you search"), and call it when another tool returns nothing, because "not in this corpus" and
        "no such holding" are different answers and only this tool can tell them apart.

        `judgments_with_text` is the number that matters. Metadata without text can resolve a citation
        and nothing else: no paragraph can be located in it, no quote checked, no proposition
        verified. `index_built` says whether a corpus-wide search can run at all.

        `courts` is here so that the limits of coverage can be stated without being guessed at. Asked
        what it holds, a model will otherwise describe the corpus it expects -- naming courts that are
        not in it, or ruling out courts that are -- and a coverage answer invented from the shape of
        the key format is still an invented answer.
        """
        with session_factory() as session:
            judgments = session.scalar(select(func.count()).select_from(Judgment)) or 0
            with_text = session.scalar(select(func.count()).select_from(JudgmentTextVersion)) or 0
            paragraphs = session.scalar(select(func.count()).select_from(Paragraph)) or 0
            indexed = search.index_exists(session)
            earliest, latest = session.execute(
                select(func.min(Judgment.decided_on), func.max(Judgment.decided_on))
            ).first() or (None, None)
            courts = session.execute(
                select(Judgment.court, func.count())
                .group_by(Judgment.court)
                .order_by(func.count().desc())
            ).all()
        return {
            "judgments": judgments,
            "judgments_with_text": with_text,
            "paragraphs": paragraphs,
            "index_built": indexed,
            "earliest_judgment": earliest,
            "latest_judgment": latest,
            "courts": {(court or "unrecorded"): count for court, count in courts},
            "note": (
                "A judgment outside this date range, or from a court not listed in `courts`, is not in "
                "the corpus, and its absence is not evidence that no such authority exists."
            ),
        }

    @tool
    def resolve_citation(citation: str) -> dict:
        """Does this case exist, and which judgment does the citation name?

        The first of the three questions the engine asks, and the only one that can be answered
        without the judgment's text. Use it when a user quotes a citation and wants to know whether it
        is real, and use it before any tool that takes a judgment key, to get that key.

        A status of "not_found" is the finding a phantom citation produces -- a case that was never
        decided, cited as if it had been. "ambiguous" means the citation names more than one judgment
        in the corpus and the candidates say which; that is a question for the user, not a coin toss.
        `name_mismatch` is the subtler one: the citation resolves, but to a case with different
        parties, which is how a real citation ends up attached to the wrong proposition.

        Args:
            citation: A citation as it was written, optionally with party names, for example
                "Kesavananda Bharati v. State of Kerala, (1973) 4 SCC 225" or "[2019] 9 S.C.R. 593".
                More than one may appear in the string; every one found is resolved.
        """
        found = extract_citations(citation)
        if not found:
            return {
                "citations": [],
                "note": (
                    "no citation was found in that string. A citation needs a reporter and a volume, "
                    "such as '(2019) 4 SCC 1' or 'INSC:2019:770'; party names alone are not one"
                ),
            }
        with session_factory() as session:
            out = []
            for parsed in found:
                result = resolve(session, parsed)
                out.append(
                    {
                        "wrote": parsed.raw,
                        "status": result.status,
                        "method": result.method,
                        "key": result.canonical_key,
                        "title": result.matched_title,
                        "score": round(result.score, 1),
                        "name_mismatch": result.name_mismatch,
                        "note": result.note,
                        "review": result.review,
                        "candidates": [
                            {
                                "key": candidate.canonical_key,
                                "title": candidate.title,
                                "score": round(candidate.score, 1),
                                "via": candidate.via,
                            }
                            for candidate in result.candidates[:3]
                        ],
                    }
                )
        return {"citations": out}

    @tool
    def check_treatment(judgment: str) -> dict:
        """Is this authority still good law? What every later judgment in the corpus did with it.

        Built from the citator's edges, not from a model's memory: each answer is the set of later
        judgments that cited this one and what they did with it, with the bench-strength rule applied,
        so a smaller bench cannot be reported as having overruled a larger one.

        Read `unchecked` before reporting `status`. "good_law" with `unchecked` true means nothing in
        the corpus has ever cited this judgment -- good law by default, not by evidence -- and saying
        "we checked and it is sound" about it would be false. Say that nothing was found against it.

        Args:
            judgment: A canonical key such as "INSC:2019:770", or a citation, which will be resolved
                first.
        """
        with session_factory() as session:
            record = _judgment_from(session, judgment)
            if record is None:
                return {"error": f"no judgment in this corpus matches {judgment!r}"}
            report = treatment_of(session, record.id)
            payload = _treatment_json(report)
        payload["title"] = record.title
        payload["key"] = record.canonical_key
        return payload

    @tool
    def locate_paragraph(judgment: str, proposition: str, claimed_pinpoint: str = "") -> dict:
        """Which paragraph of one judgment carries this proposition, and does the claimed one?

        The second question. Use it when the case is already known and what is in doubt is the
        pinpoint -- "the brief cites para 73 for this; is that right?" -- or when a user wants the
        paragraph that actually says a thing.

        `pinpoint.status` is the answer to the claim: "ok" means the paragraph exists and the words
        are in it, "wrong_paragraph" means it exists but the proposition is elsewhere in the judgment,
        "out_of_range" and "not_in_judgment" mean the cited paragraph is not there at all.
        `likely_quoted` on a candidate warns that the paragraph is a block quoted from another
        judgment, so the words are some other court's and attributing them to this one is an error.

        Args:
            judgment: A canonical key such as "INSC:2019:770", or a citation to resolve first.
            proposition: The claim to find, in the words it was written in. A sentence, not keywords.
            claimed_pinpoint: The paragraph the brief cited, if it cited one, such as "73". Leave it
                empty when nothing was claimed; the ranking then runs without a claim to check.
        """
        with session_factory() as session:
            record = _judgment_from(session, judgment)
            if record is None:
                return {"error": f"no judgment in this corpus matches {judgment!r}"}
            paragraphs = load_paragraphs(session, record.id)
            if not paragraphs:
                return {"key": record.canonical_key, "title": record.title, "error": NO_TEXT}
            result = locate(paragraphs, proposition, claimed_pinpoint=claimed_pinpoint.strip() or None)
            payload = _location_json(result)
        payload["key"] = record.canonical_key
        payload["title"] = record.title
        payload["proposition"] = proposition
        return payload

    @tool
    def verify_brief(text: str, facts: str = "") -> dict:
        """Check every citation in a passage: does it exist, which paragraph, does it support the claim.

        The whole engine, on a piece of writing. Use it when a user pastes a brief, a memorial, a
        written submission, or any passage with citations in it, and wants to know what is wrong.
        Prefer it over calling the single checks one by one: it attributes a proposition to each
        citation, runs all eight stages in a fixed order, and reports findings by the mode number from
        the taxonomy in docs/PRD.md, which is how a reader checks the engine rather than trusting it.

        This is the expensive tool. It is minutes, not seconds, on a brief with thirty citations,
        because each one that resolves has its paragraph located and read. Call it once per passage.

        Read `grade` per citation, and read `support` honestly: "not assessed" means a check could not
        be run, not that the citation passed. `needs_review` with `review_reason` marks the ones a
        human has to look at.

        Args:
            text: The passage to check, with its citations in place, as written.
            facts: The facts of the present matter, if the user gave any. With them the engine also
                asks whether each authority applies to this case rather than merely saying what the
                user claims; without them that question is skipped rather than guessed.
        """
        passage = text.strip()
        if not passage:
            return {"error": "there is nothing to check"}
        matter_facts = facts.strip()
        with session_factory() as session:
            verdicts = verify_text(
                session,
                passage,
                build_structured(ScopeAssessment),
                voice_model=build_structured(VoiceAssessment),
                weight_model=build_structured(WeightAssessment),
                facts_model=build_structured(ApplicabilityAssessment) if matter_facts else None,
                matter_facts=matter_facts,
            )
        graded: dict[str, int] = {}
        for verdict in verdicts:
            graded[verdict.grade] = graded.get(verdict.grade, 0) + 1
        return {
            "citations_found": len(verdicts),
            "by_grade": graded,
            "applicability_checked": bool(matter_facts),
            "verdicts": [_verdict_json(v) for v in verdicts],
        }

    @tool
    def find_authority(proposition: str, top: int = 5) -> dict:
        """Which judgment backs this proposition, and which line of it.

        The other direction: a proposition in, judgments and the sentence out. Use it when a user
        needs authority for something they intend to argue.

        Paragraphs reciting counsel's argument are dropped before ranking, and that is what separates
        this from a text search. An advocate states a rule more baldly than a court ever will, so a
        paragraph summarising what counsel urged matches a proposition's words better than the holding
        does -- and offering it as authority would hand the user the exact mistake `verify_brief`
        exists to catch. What comes back is the court's own voice.

        Quote `line`, cite `pinpoint`, and check `doubtful` before offering anything: an authority from
        a judgment later benches have doubted is worse than none.

        Args:
            proposition: The claim to find authority for, written as a sentence.
            top: How many authorities to return. Five is usually enough; more is slower, not better.
        """
        claim = proposition.strip()
        if not claim:
            return {"error": "there is nothing to search for"}
        with session_factory() as session:
            if not search.index_exists(session):
                return {"error": "the full-text index has not been built, so the corpus cannot be searched"}
            found = search.find_authorities(session, claim, top=max(1, min(top, 10)))
        return {
            "proposition": claim,
            "authorities": [_authority_json(a) for a in found],
            "note": None
            if found
            else (
                "no paragraph in this corpus carries these words. Call corpus_status before "
                "concluding anything: the corpus may simply not hold the years or courts in question"
            ),
        }

    @tool
    def find_contrary_authority(proposition: str, top: int = 5) -> dict:
        """Which judgment says the opposite of this proposition -- what opposing counsel will cite.

        Run this on anything a user intends to argue, unprompted if they are drafting. A brief that
        survives its own citation check still loses to the judgment it never looked for.

        Every result is a lead, not a holding, and the wording matters. `confirmed` is null when
        nobody has read the passage against the proposition, true when a model read it and found it
        genuinely opposite, false when a model read it and found it was not. Null is not false. Report
        a lead as "a passage to check", and only a confirmed one as a contradiction.

        Args:
            proposition: The claim to attack, written as a sentence.
            top: How many leads to return.
        """
        claim = proposition.strip()
        if not claim:
            return {"error": "there is nothing to search for"}
        with session_factory() as session:
            report = find_contrary(session, claim, top=max(1, min(top, 10)))
            return _contrary_json(report)

    @tool
    def bind_proposition(proposition: str) -> dict:
        """May this proposition be put in a document, behind which authority? The drafting gate.

        The strictest tool here, and the one to use when a user is writing rather than checking.
        Candidates are taken in the order the search ranked them and the first that passes the gate
        wins; each authority is verified against the proposition before it is allowed through, so
        nothing reaches a draft that `verify_brief` would have flagged in it.

        `status` is the answer. "bound" means an authority was found and verified. "narrowed" means an
        authority supports less than the proposition claims and `chosen.reason` says what it does
        support -- offer the narrowed wording, not the original. "refused" means nothing in the corpus
        may be put behind this sentence, and `considered` says what was looked at and why each was
        rejected. A refusal that names the near miss is the useful answer; do not talk around it, and
        never substitute a citation of your own.

        Args:
            proposition: The sentence the user wants to write, as they want to write it.
        """
        claim = proposition.strip()
        if not claim:
            return {"error": "there is nothing to bind"}
        with session_factory() as session:
            binding = authority_gate.bind_proposition(session, claim, build_structured(ScopeAssessment))
            return {
                "proposition": binding.proposition,
                "status": binding.status,
                "reason": binding.reason,
                "chosen": None
                if binding.chosen is None
                else {
                    "authority": _authority_json(binding.chosen.authority),
                    "status": binding.chosen.status,
                    "reason": binding.chosen.reason,
                    "verdict": None
                    if binding.chosen.verdict is None
                    else _verdict_json(binding.chosen.verdict),
                },
                "considered": [
                    {
                        "pinpoint": entry.authority.pinpoint,
                        "title": entry.authority.title,
                        "status": entry.status,
                        "reason": entry.reason,
                    }
                    for entry in binding.considered
                ],
            }

    return [
        corpus_status,
        resolve_citation,
        check_treatment,
        locate_paragraph,
        verify_brief,
        find_authority,
        find_contrary_authority,
        bind_proposition,
    ]
