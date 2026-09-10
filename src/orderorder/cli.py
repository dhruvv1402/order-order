"""Command line for OrderOrder.

The engine runs in two directions and both are here, along with everything needed to fill the
knowledge base they run against and to measure what they do.

Setting up:

    orderorder doctor [--probe]          check the environment, and that a model returns a schema
    orderorder init-db                   create tables
    orderorder corpus years|fetch        what the open-data bucket holds, and get it
    orderorder ingest metadata 2019      import a year into judgment + citation_alias
    orderorder ingest text KEY...        fetch, clean and segment a judgment's official PDF
    orderorder ingest bulk-text          the same for the whole corpus; resumable
    orderorder ingest aliases            learn the reporter citations the open data omits
    orderorder ingest repair-trailers    cut the reporter's own words out of stored text
    orderorder index                     full-text index over every paragraph
    orderorder citator                   who cited whom, and what they did with it
    orderorder stats                     row counts

Checking a citation somebody wrote:

    orderorder cite parse "..."          what the citation grammar finds in a string
    orderorder resolve "(2019) 4 SCC 1"  which judgment a citation names
    orderorder locate KEY "..."          which paragraph of one judgment carries a claim
    orderorder treatment KEY             is this authority still good law
    orderorder verify --file brief.txt   every citation in a brief, with --memo, --report, --annotate

Finding one nobody wrote yet:

    orderorder find "..."                which judgment backs a proposition, and which line
    orderorder contrary "..."            which judgment says the opposite, which is what you will meet
    orderorder argue propositions.txt    bind each proposition to an authority, or refuse to
    orderorder draft plan.txt --docx x   assemble a written submission from a case plan

Measuring both:

    orderorder eval generate             plant known failures in real judgments
    orderorder eval run [--no-model]     score the engine against them
    orderorder eval paraphrase           build restated queries for the search evaluation
    orderorder eval search               score the search direction
    orderorder eval gate                 what the drafting gate lets through, and throws out
    orderorder eval contrary             the same holding put both ways, and what comes back
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import func, select
from sqlalchemy import text as sql_text

from orderorder import __version__, logs
from orderorder.citations.grammar import extract_citations
from orderorder.config import get_settings
from orderorder.db.models import CitationAlias, Judgment, JudgmentTextVersion, Paragraph
from orderorder.db.session import current_revision, get_session, init_db, stamp_head, upgrade_db
from orderorder.drafting.assemble import assemble
from orderorder.drafting.attack import attack_draft
from orderorder.drafting.plan import PlanError, read_plan
from orderorder.drafting.render import to_markdown
from orderorder.drafting.word import write_docx
from orderorder.engine import authority, chroma_store, citator, contrary, embeddings, search
from orderorder.engine.graph import verify_text
from orderorder.engine.locator import locate as locate_claim
from orderorder.engine.memo import render_memo, write_memo
from orderorder.engine.prompts import SCOPE_PROMPT, format_candidates
from orderorder.engine.providers import build_structured, describe_providers
from orderorder.engine.quotes import find_quote
from orderorder.engine.report import annotate_brief, verification_report
from orderorder.engine.schemas import (
    ApplicabilityAssessment,
    ChallengeAssessment,
    OppositionAssessment,
    Restatement,
    ScopeAssessment,
    VoiceAssessment,
    WeightAssessment,
)
from orderorder.engine.verdict import GRADES
from orderorder.evaluation import contrary as contrary_eval
from orderorder.evaluation import gate as gate_eval
from orderorder.evaluation import retrieval
from orderorder.evaluation.generate import generate as generate_gold
from orderorder.evaluation.gold import read_gold, write_gold
from orderorder.evaluation.gold import summarise as summarise_gold
from orderorder.evaluation.run import format_disagreements, format_report, run_gold
from orderorder.ingest import aliases as alias_learning
from orderorder.ingest import bulk, repair
from orderorder.ingest import corpus as corpus_mod
from orderorder.ingest import pdf as pdf_mod
from orderorder.ingest import roles as roles_mod
from orderorder.ingest.metadata import import_parquet
from orderorder.ingest.store import load_paragraphs, store_extracted
from orderorder.resolver import resolve as resolve_citation

app = typer.Typer(help="Citation-integrity engine for Indian case law.", no_args_is_help=True)
corpus_app = typer.Typer(help="Download judgments from the AWS Open Data bucket.", no_args_is_help=True)
ingest_app = typer.Typer(help="Import downloaded data into the knowledge base.", no_args_is_help=True)
cite_app = typer.Typer(help="Citation grammar tools.", no_args_is_help=True)
eval_app = typer.Typer(help="Measure the engine against a gold set.", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")
app.add_typer(ingest_app, name="ingest")
app.add_typer(cite_app, name="cite")
app.add_typer(eval_app, name="eval")

console = Console()


@app.callback()
def _root(version: bool = typer.Option(False, "--version", help="Print the version and exit.")) -> None:
    if version:
        console.print(f"orderorder {__version__}")
        raise typer.Exit()


@app.command()
def doctor(
    probe: bool = typer.Option(
        False, "--probe", help="Make one real call to check the model returns a filled schema."
    ),
) -> None:
    """Check Python, the data directory, the database and which model providers have keys."""
    settings = get_settings()
    table = Table(title="OrderOrder environment", show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Value")
    table.add_column("Status")

    table.add_row("python", sys.version.split()[0], "ok" if sys.version_info >= (3, 12) else "needs 3.12+")
    data_dir = settings.orderorder_data_dir
    table.add_row("data dir", str(data_dir), "ok" if data_dir.exists() else "missing")
    table.add_row(
        "database", settings.db_url.split("://")[0], "postgres" if settings.is_postgres else "sqlite"
    )

    table.add_row("indian kanoon", "set" if os.environ.get("INDIANKANOON_TOKEN") else "-", "")
    console.print(table)

    providers = Table(title="Language models", show_header=True, header_style="bold")
    providers.add_column("provider:model")
    providers.add_column("key")
    # "configured", not "usable". The column is read from environment variables and says nothing
    # about whether anything answers: a key for a proxy that is not running reads exactly the same as
    # a working provider. A night was spent on that, so the word is the honest one and the line below
    # says where to find out.
    providers.add_column("configured")
    usable = 0
    for spec, env, ok in describe_providers():
        providers.add_row(spec, env, "[green]yes[/green]" if ok else "[dim]no key[/dim]")
        usable += int(ok)
    console.print(providers)
    if usable == 0:
        console.print(
            "[yellow]no model available[/yellow]: existence, pinpoint and retrieval still work; "
            "extent of support does not. Add a free key, see .env.example."
        )
    else:
        console.print(f"[green]{usable} provider(s) configured[/green]")
        if not probe:
            console.print(
                "[dim]configured means a key is set, not that anything answered. "
                "Run [bold]--probe[/bold] to make one real call and find out.[/dim]"
            )

    if probe:
        _probe_structured_output()

    try:
        with get_session() as session:
            judgments = session.scalar(select(func.count()).select_from(Judgment)) or 0
        console.print(f"[green]database reachable[/green], {judgments} judgments")
    except Exception as exc:  # noqa: BLE001 - the point is to report any failure
        console.print(f"[yellow]database not ready:[/yellow] {exc}")
        console.print("run [bold]orderorder init-db[/bold]")


def _probe_structured_output() -> None:
    """Ask the configured provider one question whose answer is checkable, and report what came back.

    Every model call in this engine is schema-bound, and an endpoint that accepts the request but
    answers in prose is the failure that matters: nothing crashes, and every extent-of-support check
    quietly becomes "not assessed". A gateway or a self-hosted server may or may not pass schema
    decoding through to whichever model it routed to, and the only way to know is to ask it.
    """
    model = build_structured(ScopeAssessment)
    if model is None:
        console.print("[yellow]nothing to probe[/yellow]: no provider is usable")
        return

    paragraph = (
        "A misrepresentation vitiates consent only where it induced the contract, and the burden of "
        "proving inducement lies upon the party alleging it."
    )
    prompt = SCOPE_PROMPT.format(
        claim="A misrepresentation vitiates consent where it induced the contract.",
        candidates=format_candidates([("3", paragraph)]),
    )
    try:
        answer = model.invoke(prompt)
    except Exception as exc:  # noqa: BLE001 - the point is to report whatever went wrong
        console.print(f"[red]probe failed[/red]: {type(exc).__name__}: {exc}")
        return

    if not isinstance(answer, ScopeAssessment):
        console.print(
            f"[red]probe returned {type(answer).__name__}[/red], not a filled schema. This endpoint "
            "accepts the request but does not constrain the answer, so extent of support, weight and "
            "applicability would all report 'not assessed'."
        )
        return

    console.print(f"[green]schema returned[/green]: support={answer.support!r} para={answer.paragraph_label!r}")
    if answer.quote:
        verified = find_quote(answer.quote, paragraph).found
        colour = "green" if verified else "yellow"
        console.print(f"  [{colour}]quote verifies against the text: {verified}[/{colour}]")
    else:
        console.print("  [yellow]no quote returned[/yellow]; a claim of support could not be grounded")


@app.command("init-db")
def init_db_command() -> None:
    """Create tables in the configured database, and record which schema they are.

    The stamp is the half that matters later. A database created without one looks, to `migrate`,
    like a database that has never had anything applied to it, so the next release would try to
    create tables that are already there and stop. Stamping at creation means the first real
    migration, whenever it comes, applies to this database cleanly.
    """
    settings = get_settings()
    init_db()
    stamp_head()
    console.print(f"[green]tables created[/green] in {settings.db_url}")
    console.print(f"schema stamped at [bold]{current_revision() or 'none'}[/bold]")


@app.command("migrate")
def migrate_command(
    stamp: bool = typer.Option(
        False,
        "--stamp",
        help="Record the latest revision without applying anything. For a database whose tables "
        "already exist -- an ingested corpus, or one built before migrations.",
    ),
) -> None:
    """Bring the configured database up to the current schema.

    `init_db` creates missing tables and never alters an existing one, so it cannot be how a deployed
    database changes shape: the first column this application wants to widen would otherwise be a
    manual job on a 1.2 GB file with the corpus in it. This is the other half.

    On an **empty** database, run it plain: the baseline builds every table.
    On a database that **already holds the corpus**, run it once with `--stamp`. The tables are there;
    stamping records that the baseline is applied, so that the next migration is the only thing that
    runs against it.
    """
    settings = get_settings()
    before = current_revision()
    if stamp:
        stamp_head()
        console.print(f"[green]stamped[/green] {settings.db_url} at [bold]{current_revision()}[/bold]")
        console.print("nothing was applied; the schema was taken as already current")
        return

    upgrade_db()
    after = current_revision()
    if before == after:
        console.print(f"[green]already current[/green] at [bold]{after}[/bold]")
    else:
        console.print(f"[green]migrated[/green] {before or 'nothing'} → [bold]{after}[/bold]")


@corpus_app.command("years")
def corpus_years() -> None:
    """List the years available in the open-data bucket."""
    years = corpus_mod.available_years()
    console.print(f"{len(years)} years: {years[0]}-{years[-1]}")
    console.print(f"[dim]{corpus_mod.ATTRIBUTION}[/dim]")


@corpus_app.command("fetch")
def corpus_fetch(
    years: list[int] = typer.Argument(..., help="Years to download, e.g. 2019 2020 2021."),
) -> None:
    """Download parquet metadata for the given years."""
    settings = get_settings()
    for year in years:
        path = corpus_mod.download_metadata(year, settings.corpus_dir)
        size_mb = path.stat().st_size / 1e6
        console.print(f"[green]{year}[/green] {path.name} ({size_mb:.1f} MB)")
    console.print(f"[dim]{corpus_mod.ATTRIBUTION}[/dim]")


@ingest_app.command("metadata")
def ingest_metadata(
    years: list[int] = typer.Argument(..., help="Years to import; download them first."),
    limit: int | None = typer.Option(None, help="Import only the first N rows of each year."),
) -> None:
    """Import parquet metadata into judgment and citation_alias."""
    settings = get_settings()
    init_db()
    totals: dict[str, int] = {}
    for year in years:
        path = settings.corpus_dir / "metadata" / f"year={year}_metadata.parquet"
        if not path.exists():
            path = corpus_mod.download_metadata(year, settings.corpus_dir)
        with get_session() as session:
            stats = import_parquet(session, path, limit=limit)
        console.print(f"[green]{year}[/green] {stats.as_dict()}")
        for key, value in stats.as_dict().items():
            totals[key] = totals.get(key, 0) + value
    if len(years) > 1:
        console.print(f"[bold]total[/bold] {totals}")


@ingest_app.command("bulk-text")
def ingest_bulk_text(
    years: list[int] = typer.Option(None, "--year", help="Restrict to these years; repeatable."),
    limit: int | None = typer.Option(None, help="Stop after this many judgments."),
    workers: int = typer.Option(bulk.DEFAULT_WORKERS, help="Concurrent PDF fetches."),
    version_key: str = typer.Option("scr_pdf", help="Name for this text version."),
    retry: bool = typer.Option(False, help="Retry only the judgments in the last failures file."),
) -> None:
    """Fetch and store text for every judgment that does not have it yet.

    Resumable: judgments already holding this text version are skipped, and PDFs already on disk are
    not downloaded again, so a run that dies part-way carries on where it stopped.
    """
    settings = get_settings()
    init_db()
    failures_path = settings.corpus_dir / "text_ingest_failures.tsv"

    with get_session() as session:
        if retry:
            keys = list(bulk.read_failures(failures_path))
            if not keys:
                console.print(f"[yellow]no failures recorded[/yellow] in {failures_path}")
                raise typer.Exit(1)
            judgments = list(
                session.scalars(select(Judgment).where(Judgment.canonical_key.in_(keys))).all()
            )
        else:
            judgments = bulk.judgments_needing_text(
                session, version_key=version_key, years=list(years) if years else None, limit=limit
            )

        already = session.scalar(
            select(func.count()).select_from(JudgmentTextVersion).where(
                JudgmentTextVersion.version_key == version_key
            )
        ) or 0
        if not judgments:
            console.print(f"[green]nothing to do[/green]: {already} judgments already have text")
            return

        console.print(
            f"fetching text for [bold]{len(judgments)}[/bold] judgments "
            f"({already} already done) with {workers} workers"
        )

        def report(outcome: bulk.Outcome, done: int, total: int) -> None:
            if not outcome.ok:
                console.print(f"  [yellow]{outcome.canonical_key}[/yellow] {outcome.status}: {outcome.detail}")
            elif done % 50 == 0 or done == total:
                console.print(f"  [dim]{done}/{total}[/dim] {outcome.canonical_key} {outcome.paragraphs} paras")

        result = bulk.ingest_text_bulk(
            session, judgments, settings.corpus_dir, version_key=version_key,
            workers=workers, on_result=report,
        )

    console.print(f"[bold]done[/bold] {result.counts()}, {result.paragraphs:,} paragraphs stored")
    written = bulk.write_failures(result, failures_path)
    if written:
        console.print(
            f"[yellow]{written} failures[/yellow] written to {failures_path}; "
            "retry them with [bold]orderorder ingest bulk-text --retry[/bold]"
        )


@ingest_app.command("repair-trailers")
def ingest_repair_trailers(
    dry_run: bool = typer.Option(False, help="Report what would be removed without writing it."),
) -> None:
    """Cut the reporter's closing matter out of judgments ingested before extraction knew to.

    The SCR volumes end a judgment with the editors' own words, running on from the court's last
    paragraph. Extraction now stops at them; this removes them from what was stored earlier, without
    re-reading nine thousand PDFs.

    This rewrites paragraph bodies under the same ids, which is the one change `orderorder index`
    cannot detect: nothing about the row's identity changes, so reconciling finds nothing to do and
    the index keeps serving the publisher's words as the court's. Rebuild it afterwards.
    """
    init_db()
    prefix = "[yellow]would remove[/yellow]" if dry_run else "[green]removed[/green]"
    with get_session() as session:
        console.print(f"{prefix}: {repair.strip_publisher_trailers(session, dry_run=dry_run)}")
        console.print(f"{prefix}: {repair.strip_signoffs(session, dry_run=dry_run)}")
    if not dry_run:
        console.print(
            "[yellow]the full-text index is now stale[/yellow]: this rewrote paragraphs in place, "
            "which reconciling cannot see. Run [bold]orderorder index --rebuild[/bold]."
        )


@ingest_app.command("mark-opinions")
def ingest_mark_opinions(
    dry_run: bool = typer.Option(False, help="Report what would be marked without writing it."),
) -> None:
    """Find the dissents and concurrences that only announce themselves in a sentence.

    A printed "(dissenting)" appears five times in nine thousand judgments. What a dissenting judge
    actually writes is "I regret my inability to agree", and extraction now reads that; this applies
    the corrected rule to text already stored.
    """
    init_db()
    with get_session() as session:
        result = repair.mark_separate_opinions(session, dry_run=dry_run)
    prefix = "[yellow]would mark[/yellow]" if dry_run else "[green]marked[/green]"
    console.print(f"{prefix}: {result}")


@ingest_app.command("mark-roles")
def ingest_mark_roles(
    relabel: bool = typer.Option(False, help="Re-classify paragraphs that already carry a role."),
    dry_run: bool = typer.Option(False, help="Report what would be labelled without writing it."),
) -> None:
    """Give every stored paragraph its rhetorical role (OpenNyAI label set, cue classifier).

    The weight and voice checks only mean something if the paragraph being tested is the court's own
    reasoning; `ratio`, `precedent_relied` and `precedent_not_relied` make that knowledge searchable.
    Without a role label, retrieval cannot tell the facts paragraph from the holding paragraph, and
    the answer to a legal question gets quotes from case history. Paragraphs already carrying a role
    are left alone unless --relabel: the printed label is the better evidence, and re-running this
    must stay idempotent.
    """
    init_db()
    with get_session() as session:
        result = roles_mod.mark_roles(session, dry_run=dry_run, relabel=relabel)
    prefix = "[yellow]would label[/yellow]" if dry_run else "[green]labelled[/green]"
    console.print(f"{prefix}: {result}")
    if result.skipped_already:
        console.print(f"[dim]skipped, already labelled: {result.skipped_already:,}[/dim]")


@ingest_app.command("aliases")
def ingest_aliases(
    limit: int | None = typer.Option(None, help="Read only this many judgments."),
    dry_run: bool = typer.Option(False, help="Report what would be learned without writing it."),
) -> None:
    """Learn the reporter citations the open data does not carry, from how judgments cite each other.

    The metadata gives every judgment its neutral and SCR citations. Practice runs on SCC, so a brief
    citing a real case by its SCC number would otherwise resolve to nothing and be called a phantom.
    """
    init_db()
    with get_session() as session:

        def progress(done: int, total: int) -> None:
            if done % 1000 == 0 or done == total:
                console.print(f"  [dim]{done}/{total}[/dim] judgments read")

        before = session.scalar(select(func.count()).select_from(CitationAlias)) or 0
        stats = alias_learning.learn_aliases(
            session, limit=limit, on_progress=progress, dry_run=dry_run
        )
        after = session.scalar(select(func.count()).select_from(CitationAlias)) or 0

    console.print(f"[bold]{stats.as_dict()}[/bold]")
    console.print(f"[dim]inferred by reporter: {stats.by_reporter}[/dim]")
    if dry_run:
        console.print("[yellow]dry run[/yellow]: nothing written")
    else:
        console.print(f"[green]aliases {before:,} -> {after:,}[/green] (+{after - before:,})")


@ingest_app.command("text")
def ingest_text(
    keys: list[str] = typer.Argument(..., help="Canonical keys, e.g. INSC:2019:770."),
    version_key: str = typer.Option("scr_pdf", help="Name for this text version."),
) -> None:
    """Fetch each judgment's official PDF, clean it, segment it and store the paragraphs."""
    settings = get_settings()
    init_db()
    for key in keys:
        with get_session() as session:
            judgment = session.scalars(select(Judgment).where(Judgment.canonical_key == key)).first()
            if judgment is None:
                console.print(f"[red]{key}[/red] not in the knowledge base; import its year first")
                continue
            if not judgment.source_id:
                console.print(f"[red]{key}[/red] has no source path, cannot locate a PDF")
                continue
            year = (judgment.extra or {}).get("year") or (
                judgment.decided_on.year if judgment.decided_on else None
            )
            try:
                path = pdf_mod.fetch_pdf(judgment.source_id, year, settings.corpus_dir)
                extracted = pdf_mod.extract(path, judgment.source_id)
            except Exception as exc:  # noqa: BLE001 - report and continue with the next key
                console.print(f"[red]{key}[/red] {type(exc).__name__}: {exc}")
                continue
            if not extracted.has_judgment:
                console.print(f"[yellow]{key}[/yellow] no judgment text found after cleaning")
                continue
            result = store_extracted(
                session,
                judgment,
                extracted,
                version_key=version_key,
                source_url=pdf_mod.pdf_url(judgment.source_id, year),
            )
            bench = f"bench {extracted.bench_strength}" if extracted.bench_strength else "bench ?"
            console.print(
                f"[green]{key}[/green] {extracted.page_count}p "
                f"{len(extracted.judgment):,}ch {result.paragraphs} paras, "
                f"{bench}, author {extracted.author or '-'}"
                + (" [yellow](bench corrected)[/yellow]" if result.bench_corrected else "")
                + (" [dim](replaced)[/dim]" if result.replaced else "")
            )


def _law_badge(verdict) -> str:
    """Whether the authority is still good law, in one word."""
    if verdict.treatment is None:
        return "-"
    if verdict.treatment.is_undermined:
        return "[yellow]undermined[/yellow]"
    if verdict.treatment.is_doubtful:
        return f"[red]{verdict.treatment.status.replace('_', ' ')}[/red]"
    if verdict.treatment.citing_count == 0:
        return "[dim]uncited[/dim]"
    return f"[green]good[/green] [dim]({verdict.treatment.citing_count})[/dim]"


def _shorten(text: str, limit: int) -> str:
    """Trim to a whole word. A claim cut mid-word reads as though the engine misread the brief."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + " ..."


VOICE_BADGES = {
    "court_majority": "court",
    "court_concurring": "concurring",
    "court_dissent": "[red]dissent[/red]",
    "counsel_argument": "[red]counsel[/red]",
    "lower_court": "[red]court below[/red]",
    "quoted_precedent": "[yellow]quoted[/yellow]",
    "headnote": "[red]headnote[/red]",
    "unclear": "[dim]unclear[/dim]",
}


def _voice_badge(verdict) -> str:
    """One word for whose passage it is. A dash means no paragraph was definite enough to attribute."""
    if verdict.voice is None:
        return "-"
    badge = VOICE_BADGES.get(verdict.voice.voice, verdict.voice.voice)
    if verdict.weight is not None and verdict.weight.is_obiter:
        badge += " [yellow]obiter[/yellow]"
    return badge


@app.command("locate")
def locate_command(
    key: str = typer.Argument(..., help="Canonical key of the judgment, e.g. INSC:2019:770."),
    proposition: str = typer.Argument(..., help="The proposition the brief attributes to the judgment."),
    pinpoint: str | None = typer.Option(None, help="The paragraph the brief cited, e.g. 5.1."),
    top: int = typer.Option(5, help="How many candidate paragraphs to show."),
) -> None:
    """Rank a judgment's paragraphs against a proposition and check the pinpoint."""
    with get_session() as session:
        judgment = session.scalars(select(Judgment).where(Judgment.canonical_key == key)).first()
        if judgment is None:
            console.print(f"[red]{key}[/red] not in the knowledge base")
            raise typer.Exit(1)
        paragraphs = load_paragraphs(session, judgment.id)
        if not paragraphs:
            console.print(
                f"[yellow]{key}[/yellow] has no text; run [bold]orderorder ingest text {key}[/bold]"
            )
            raise typer.Exit(1)

        result = locate_claim(paragraphs, proposition, claimed_pinpoint=pinpoint, top_k=top)
        console.print(f"[bold]{judgment.title[:80]}[/bold]")
        console.print(
            f"[dim]{len(paragraphs)} paragraphs; query terms: {', '.join(result.query_terms[:12])}[/dim]\n"
        )

        check = result.pinpoint
        if check.status == "ok":
            console.print(f"[green]pinpoint ok[/green] paragraph {check.claimed} exists")
        elif check.is_problem:
            console.print(f"[red]pinpoint problem[/red] {check.note}")
        elif check.status == "none_claimed":
            console.print(f"[dim]no pinpoint claimed; numbering runs to {check.highest_label}[/dim]")

        table = Table(show_header=True, header_style="bold")
        table.add_column("para")
        table.add_column("opinion")
        table.add_column("score", justify="right")
        table.add_column("terms")
        table.add_column("text")
        for candidate in result.candidates[:top]:
            label = candidate.printed_label or f"#{candidate.seq}"
            if candidate.is_claimed_pinpoint:
                label = f"{label} (cited)"
            if candidate.likely_quoted:
                label = f"[red]{label} quoted?[/red]"
            opinion = candidate.opinion_kind or "-"
            if opinion == "dissenting":
                opinion = "[red]dissenting[/red]"
            table.add_row(
                label,
                opinion,
                f"{candidate.score:.2f}",
                ", ".join(candidate.matched_terms[:4]),
                candidate.preview,
            )
        console.print(table)
        if any(c.likely_quoted for c in result.candidates[:top]):
            console.print(
                "[red]warning[/red] a candidate's paragraph number breaks the judgment's sequence, "
                "so it is probably quoted from another judgment rather than this court's own words"
            )


@eval_app.command("generate")
def eval_generate(
    out: str = typer.Option("evals/gold.jsonl", help="Where to write the gold set."),
    seeds: int = typer.Option(8, help="How many judgments to plant errors in."),
    rng_seed: int = typer.Option(
        20260904,
        help="Which judgments get drawn. Change it for a held-out set the detectors were not tuned on.",
    ),
) -> None:
    """Build a gold set by planting known failure modes in real judgments."""
    init_db()
    with get_session() as session:
        items = generate_gold(session, seeds=seeds, seed_value=rng_seed)
    path = Path(out)
    write_gold(items, path)
    console.print(f"[green]{len(items)} items[/green] written to {path}")
    table = Table(show_header=True, header_style="bold")
    table.add_column("planted")
    table.add_column("items", justify="right")
    for name, count in summarise_gold(items).items():
        table.add_row(name, str(count))
    console.print(table)


@eval_app.command("run")
def eval_run(
    gold: str = typer.Option("evals/gold.jsonl", help="Gold set to score against."),
    report: str | None = typer.Option(None, help="Also write the report to this file."),
    limit: int | None = typer.Option(None, help="Score only the first N items."),
    no_model: bool = typer.Option(
        False,
        "--no-model",
        help="Score only the checks that need no model. Seconds instead of an hour.",
    ),
    detail: bool = typer.Option(
        False, "--detail", help="Print every item the engine disagreed with the gold label about."
    ),
    pace: float = typer.Option(
        0.0, help="Seconds between items. Use it on a free tier: 12 keeps two calls under 10/minute."
    ),
) -> None:
    """Run the engine over the gold set and report what it caught and what it invented.

    On a rate-limited provider, pass `--pace`. Without it the quota is exhausted in the first minute,
    every call after that is a 429, the fallback chain answers them with the next provider in the
    list, and the report describes a blend of two models rather than either one.
    """
    items = read_gold(Path(gold))
    if not items:
        console.print(f"[red]no gold set at {gold}[/red]; run [bold]orderorder eval generate[/bold]")
        raise typer.Exit(1)
    if limit:
        items = items[:limit]

    # The model-free score is the one to watch while developing the detectors: it runs in seconds
    # rather than an hour, and every check it covers is one that costs nothing per citation.
    model = None if no_model else build_structured(ScopeAssessment)
    if model is None:
        console.print(
            "[yellow]no language model configured[/yellow]: the modes that need one are reported as "
            "unassessed rather than counted as misses."
        )

    def progress(result, done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            console.print(f"  [dim]{done}/{total}[/dim]")

    with get_session() as session:
        scored = run_gold(
            session,
            items,
            model,
            voice_model=None if no_model else build_structured(VoiceAssessment),
            weight_model=None if no_model else build_structured(WeightAssessment),
            facts_model=None if no_model else build_structured(ApplicabilityAssessment),
            pace=pace,
            on_result=progress,
        )

    lines = format_report(scored)
    if detail:
        lines += ["", *format_disagreements(scored)]
    console.print("\n".join(lines))
    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")
        console.print(f"[dim]written to {report}[/dim]")


@eval_app.command("paraphrase")
def eval_paraphrase(
    out: str = typer.Option("evals/paraphrases.jsonl", help="Where to write the query set."),
    judgments: int = typer.Option(40, help="How many judgments to draw propositions from."),
    per_judgment: int = typer.Option(2, help="Propositions per judgment."),
    rng_seed: int = typer.Option(20260904, help="Which judgments get drawn."),
) -> None:
    """Build paraphrase queries for the search evaluation, and keep them.

    Every other query shape the harness builds is a run of the judgment's own text, which measures
    quoted search and says nothing about the hardest and commonest case: a lawyer's words and the
    court's sharing only the idea. Something has to do the restating, so a model does, once, and the
    queries are written to a file that anyone can read and that later runs reuse without a model.
    """
    init_db()
    model = build_structured(Restatement)
    if model is None:
        console.print("[red]no language model configured[/red]; this is the one command that needs one")
        raise typer.Exit(1)

    def progress(_item, done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            console.print(f"  [dim]{done}/{total}[/dim]")

    with get_session() as session:
        items = retrieval.build_paraphrases(
            session,
            model,
            judgments=judgments,
            per_judgment=per_judgment,
            seed_value=rng_seed,
            on_result=progress,
        )
    written = retrieval.write_items(items, Path(out))
    console.print(f"[green]{written} paraphrase queries[/green] written to {out}")
    for item in items[:3]:
        console.print("")
        console.print(f"[dim]court:[/dim] {item.sentence[:150]}")
        console.print(f"[dim]query:[/dim] {item.query[:150]}")


@eval_app.command("search")
def eval_search(
    judgments: int = typer.Option(40, help="How many judgments to draw propositions from."),
    per_judgment: int = typer.Option(2, help="Propositions per judgment."),
    top: int = typer.Option(10, help="How deep in the ranking to look for the right answer."),
    rng_seed: int = typer.Option(20260904, help="Which judgments get drawn."),
    report: str | None = typer.Option(None, help="Also write the report to this file."),
    misses: bool = typer.Option(False, "--misses", help="Print the searches that found nothing."),
    dense: bool = typer.Option(
        False, "--dense", help="Fuse the vector ranking in too. Off because it measures worse."
    ),
    queries: str | None = typer.Option(
        None, "--queries", help="A query set built earlier, such as one from `eval paraphrase`."
    ),
) -> None:
    """Measure the search direction: a proposition in, the judgment and the line out.

    Ground truth is known by construction — every proposition is a sentence the court wrote, in a
    paragraph the corpus can name — so this needs no labelling and no model.
    """
    init_db()
    with get_session() as session:
        if not search.index_exists(session):
            console.print("[red]no search index[/red]; run [bold]orderorder index[/bold] first")
            raise typer.Exit(1)
        items = (
            retrieval.read_items(Path(queries))
            if queries
            else retrieval.build_items(
                session, judgments=judgments, per_judgment=per_judgment, seed_value=rng_seed
            )
        )
        if queries and not items:
            console.print(f"[red]no queries in {queries}[/red]")
            raise typer.Exit(1)
        console.print(f"[dim]{len(items)} queries from {judgments} judgments[/dim]")

        def progress(_outcome, done: int, total: int) -> None:
            if done % 25 == 0 or done == total:
                console.print(f"  [dim]{done}/{total}[/dim]")

        scored = retrieval.run_retrieval(session, items, top=top, dense=dense, on_result=progress)

    lines = retrieval.format_retrieval(scored, top=top)
    if misses:
        lines += ["", *retrieval.format_misses(scored)]
    console.print("\n".join(lines))
    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")
        console.print(f"[dim]written to {report}[/dim]")


@eval_app.command("gate")
def eval_gate(
    judgments: int = typer.Option(40, help="How many judgments to draw propositions from."),
    per_judgment: int = typer.Option(1, help="Propositions of each kind per judgment."),
    checked: int = typer.Option(
        gate_eval.FIELD, help="How wide a field of candidates the gate sees."
    ),
    rng_seed: int = typer.Option(20260904, help="Which judgments get drawn."),
    no_model: bool = typer.Option(
        False, "--no-model", help="Run only the half of the gate that needs no model."
    ),
    report: str | None = typer.Option(None, help="Also write the report to this file."),
) -> None:
    """Measure the drafting gate: what the retriever hands up, and what the gate does with it.

    The gate is the component with the most to lose by being wrong, because it is the one that
    writes. A verification miss leaves a bad citation in somebody else's brief; a gate miss puts one
    in yours, over your signature, with a pinpoint that makes it look checked.

    Propositions are drawn from three kinds of paragraph — the court's own words, counsel's
    submission, a dissent — and put to the search the way an advocate would put them. The number the
    gate exists for is how much of what comes back is not citable, and it is not small.
    """
    init_db()
    model = None if no_model else build_structured(ScopeAssessment)
    if model is None and not no_model:
        console.print(
            "[yellow]no language model configured[/yellow], so nothing can be bound. The "
            "model-free half of the gate is still measured."
        )
    with get_session() as session:
        if not search.index_exists(session):
            console.print("[red]no search index[/red]; run [bold]orderorder index[/bold] first")
            raise typer.Exit(1)
        items = gate_eval.build_items(
            session, judgments=judgments, per_judgment=per_judgment, seed_value=rng_seed
        )
        if not items:
            console.print("[red]no propositions could be drawn[/red]; is there text in the corpus?")
            raise typer.Exit(1)
        console.print(f"[dim]{len(items)} propositions[/dim]")

        def progress(done: int, total: int) -> None:
            if done % 10 == 0 or done == total:
                console.print(f"  [dim]{done}/{total}[/dim]")

        scored = gate_eval.run_gate(session, items, model, checked=checked, on_item=progress)

    lines = gate_eval.format_gate(scored)
    console.print("\n".join(lines))
    if report:
        gate_eval.write_report(scored, Path(report))
        console.print(f"[dim]written to {report}[/dim]")


@eval_app.command("contrary")
def eval_contrary(
    judgments: int = typer.Option(40, help="How many judgments to draw holdings from."),
    per_judgment: int = typer.Option(1, help="Holdings per judgment."),
    top: int = typer.Option(5, help="How many leads the engine may return per proposition."),
    rng_seed: int = typer.Option(20260904, help="Which judgments get drawn."),
    report: str | None = typer.Option(None, help="Also write the report to this file."),
    examples: bool = typer.Option(True, help="Show the leads returned for a court's own words."),
    rule_like: bool = typer.Option(
        True, help="Draw only sentences that state law, not ones that decide a case."
    ),
    weighted: bool = typer.Option(
        True,
        "--weighted/--no-weighted",
        help="Weigh shared terms by rarity. Off counts them, which is the comparison.",
    ),
    read: int = typer.Option(
        0, "--read", help="Also have a model read this many pairs. 0 runs no model at all."
    ),
    pace: float = typer.Option(
        0.0, help="Seconds between model calls. A free tier at 10 a minute wants 7."
    ),
) -> None:
    """Measure the contrary search: the same holding put both ways, and what comes back.

    Each item is a sentence a court wrote. It is searched twice — negated, which is what the other
    side argues, and as written, which is what the side relying on it argues. The difference between
    the two is the measurement, because a tool that answers the same way to both is reading the words
    and not the sense.
    """
    init_db()
    with get_session() as session:
        if not search.index_exists(session):
            console.print("[red]no search index[/red]; run [bold]orderorder index[/bold] first")
            raise typer.Exit(1)
        items = contrary_eval.build_items(
            session,
            judgments=judgments,
            per_judgment=per_judgment,
            seed_value=rng_seed,
            rule_like=rule_like,
        )
        if not items:
            console.print("[red]no holdings could be drawn[/red]; is there text in the corpus?")
            raise typer.Exit(1)
        console.print(f"[dim]{len(items)} searches from {judgments} judgments[/dim]")

        def progress(done: int, total: int) -> None:
            if done % 10 == 0 or done == total:
                console.print(f"  [dim]{done}/{total}[/dim]")

        scored = contrary_eval.run_contrary(
            session, items, top=top, rule_like=rule_like, weighted=weighted, on_item=progress
        )

        if read:
            model = build_structured(OppositionAssessment)
            if model is None:
                console.print("[yellow]no language model configured[/yellow]; the reading is skipped")
            else:
                console.print(f"[dim]reading {read} pairs, two calls each[/dim]")

                def reading_progress(done: int, total: int) -> None:
                    console.print(f"  [dim]read {done}/{total}[/dim]")

                contrary_eval.read_pairs(
                    session, scored, model, limit=read, pace=pace, on_item=reading_progress
                )

    lines = contrary_eval.format_contrary(scored) + contrary_eval.format_reading(scored)
    if examples:
        lines += ["", *contrary_eval.format_examples(scored)]
    console.print("\n".join(lines))
    if report:
        contrary_eval.write_report(scored, Path(report), examples=examples)
        console.print(f"[dim]written to {report}[/dim]")


@app.command("serve")
def serve_command(
    host: str = typer.Option("127.0.0.1", help="Interface to listen on. Local only by default."),
    port: int = typer.Option(8000, help="Port to listen on."),
    reload: bool = typer.Option(False, help="Restart on source changes, for development."),
) -> None:
    """Open the engine in a browser: verdict board, annotated brief, judgment viewer.

    The three things a stress-test produces are worth reading together rather than one at a time,
    which is what a page adds over this command line: click a flag, see the paragraph the engine read,
    see the verified sentence highlighted inside it.

    It binds to localhost, because the corpus and any brief you paste into it stay on this machine.
    Bind it anywhere else and a token is required: everything served would otherwise be reachable by
    anything that can route here, including the upload endpoint and a model key that costs money.
    """
    import uvicorn

    from orderorder.web.auth import TOKEN_ENV, BindingRefused, check_binding

    try:
        check_binding(host, os.environ.get(TOKEN_ENV) or None)
    except BindingRefused as refused:
        console.print(f"[red]{refused}[/red]")
        raise typer.Exit(2) from refused

    init_db()
    # A model that has started refusing every call produces a page full of honest "not assessed" and
    # no other sign. `ORDERORDER_LOG_LEVEL` turns the volume up or down; this is where it starts.
    logs.configure()
    console.print(f"[green]OrderOrder[/green] on [bold]http://{host}:{port}[/bold]  (ctrl-c to stop)")
    uvicorn.run("orderorder.web.api:app", host=host, port=port, reload=reload, log_level="warning")


@app.command("index")
def index_command(
    rebuild: bool = typer.Option(
        False, help="Drop and re-tokenise everything. Needed only after text was rewritten in place."
    ),
) -> None:
    """Bring the full-text index level with the paragraphs that are stored.

    Run it after every batch of ingestion. It adds what is missing and removes what no longer belongs,
    so running it twice does nothing the second time and a judgment ingested later is picked up
    without re-tokenising the corpus.

    `--rebuild` is for the one thing reconciling cannot see: a paragraph whose text was rewritten
    under the same id, which is what `ingest repair-trailers` does.
    """
    init_db()
    with get_session() as session:
        rows = search.build_index(session, rebuild=rebuild)
    console.print(f"[green]index ready[/green]: {rows:,} paragraphs")


@app.command("embed")
def embed_command(
    model: str = typer.Option(
        embeddings.DEFAULT_MODEL,
        help="Which encoder to use. The default is static and does the corpus in minutes.",
    ),
    limit: int | None = typer.Option(None, help="Embed only the first N paragraphs, to try it out."),
    api: bool = typer.Option(
        False,
        "--api",
        help="Encode through the configured hosted endpoint (EMBEDDINGS_BASE_URL / "
        "_API_KEY / _MODEL): BAAI/bge-m3 on Bitdeer, a self-hosted Qwen3-Embedding, any "
        "/v1/embeddings server. Resumable; ~$3 for this corpus at bge-m3 prices.",
    ),
) -> None:
    """Give every paragraph a vector, so search can match meaning and not only words.

    The lexical search finds a line it has been handed word for word. Given the same proposition in an
    advocate's own words it finds the right judgment first a third of the time, because BM25 matches
    words and a paraphrase shares none. This is the other half; the two are fused, not swapped.
    """
    init_db()
    if api:
        console.print(f"[dim]endpoint: {get_settings().embeddings_base_url or '(not configured)'}[/dim]")
        console.print(f"[dim]model: {get_settings().embeddings_model}[/dim]")
        # A trial run must not overwrite the store a full run produced; it lands beside it.
        store = embeddings.default_store()
        if limit:
            from orderorder.config import get_settings as _gs

            store = embeddings.VectorStore(_gs().data_dir / "vectors-trial")
            console.print(f"[yellow]trial run[/yellow]: writing to {store.directory}, not the main store")
        with get_session() as session:
            written = embeddings.build_api(
                session,
                store=store,
                limit=limit,
                on_progress=lambda done, total: (
                    console.print(f"  [dim]{done:,}/{total:,}[/dim]")
                    if done % 20_000 < 64 or done == total
                    else None
                ),
            )
        if not written:
            console.print("[yellow]nothing to embed[/yellow]; ingest some judgment text first")
            raise typer.Exit(1)
        size = store.vectors_path.stat().st_size / 1e6
        console.print(f"[green]{written:,} paragraphs embedded[/green] into {store.vectors_path} ({size:.1f} MB)")
        if limit:
            console.print(
                "[dim]to query this trial store:[/dim] "
                "uv run python -c \"from orderorder.engine import embeddings; "
                "from orderorder.config import get_settings; "
                "from pathlib import Path; "
                "s = embeddings.VectorStore(get_settings().data_dir / 'vectors-trial'); "
                "print(embeddings.search('your query', store=s, top=3))\""
            )
        return
    console.print(f"[dim]encoder: {model}[/dim]")

    def progress(done: int, total: int) -> None:
        if done % 40_000 < embeddings.BATCH or done == total:
            console.print(f"  [dim]{done:,}/{total:,}[/dim]")

    with get_session() as session:
        written = embeddings.build(session, model_name=model, limit=limit, on_progress=progress)
    store = embeddings.default_store()
    if not written:
        console.print("[yellow]nothing to embed[/yellow]; ingest some judgment text first")
        raise typer.Exit(1)
    size = store.vectors_path.stat().st_size / 1e6
    console.print(f"[green]{written:,} paragraphs embedded[/green] into {store.vectors_path} ({size:.0f} MB)")


@app.command("citator")
def citator_command(
    limit: int | None = typer.Option(None, help="Read only this many judgments."),
    rebuild: bool = typer.Option(False, help="Drop extracted edges and start again."),
) -> None:
    """Extract citation edges from the corpus: who cited whom, and what they did with it."""
    init_db()
    with get_session() as session:

        def progress(done: int, total: int) -> None:
            if done % 500 == 0 or done == total:
                console.print(f"  [dim]{done}/{total}[/dim] judgments read")

        stats = citator.build_citator(session, limit=limit, rebuild=rebuild, on_progress=progress)
    console.print(
        f"[green]{stats.edges:,} edges[/green] from {stats.judgments_read:,} judgments\n"
        f"[dim]{stats.treatments}[/dim]"
    )


@app.command("chroma-import")
def chroma_import_command() -> None:
    """Copy the built paragraph vectors into a local Chroma database.

    Same vectors, no re-embedding: the memmap store remains what ranking reads, and this is the
    same matrix handed to a database that can answer filtered queries and be reached from outside
    the process. Re-runnable -- rows already in Chroma are skipped.
    """
    init_db()
    added = chroma_store.import_from_store(progress=_chroma_progress_printer())
    if added == 0:
        console.print(f"[green]already imported[/green]: {chroma_store.count():,} vectors in {chroma_store.chroma_dir()}")
    else:
        console.print(
            f"[green]imported {added:,} vectors[/green] into {chroma_store.chroma_dir()} "
            f"({chroma_store.count():,} total)"
        )


# One progress line per this many vectors. The import's batch is Chroma's own cap -- a few hundred
# rows at the encoder this corpus uses -- so without a milestone the run prints thousands of lines.
_CHROMA_PROGRESS_EVERY = 40_960


def _chroma_progress_printer() -> Callable[[int], None]:
    """A progress callback that reports each time the import passes another milestone.

    It carries the next milestone rather than testing the running total against a modulus: the total
    climbs by whatever the batch happens to be, so a modulus fires once per batch near the boundary
    and prints a small flurry each time instead of one line.
    """
    milestone = _CHROMA_PROGRESS_EVERY

    def report(added: int) -> None:
        nonlocal milestone
        if added >= milestone:
            console.print(f"  [dim]{added:,} imported[/dim]")
            milestone = added + _CHROMA_PROGRESS_EVERY

    return report


@app.command("chroma-search")
def chroma_search_command(
    query: list[str] = typer.Argument(..., help="What to look for, in your own words."),
    top: int = typer.Option(10, help="How many paragraphs to show."),
) -> None:
    """Query the Chroma vector database directly, without the lexical half."""
    import textwrap

    results = chroma_store.search(" ".join(query), top=top)
    if not results:
        console.print("[yellow]nothing in the database[/yellow]; run orderorder chroma-import")
        raise typer.Exit(1)
    init_db()
    with get_session() as session:
        for paragraph_id, similarity in results:
            row = session.execute(
                sql_text(
                    "SELECT p.printed_label, v.judgment_id, substr(p.body, 1, 240) "
                    "FROM paragraph p JOIN judgment_text_version v ON v.id = p.text_version_id "
                    "WHERE p.id = :pid"
                ),
                {"pid": paragraph_id},
            ).first()
            if row is None:
                continue
            label, judgment_id, body = row
            judgment = session.get(Judgment, judgment_id)
            console.print(f"[bold]{judgment.canonical_key}[/bold] ¶{label} ({similarity:.3f})")
            console.print(textwrap.fill(body.replace("\n", " "), 100))
            console.print()


@app.command("treatment")
def treatment_command(
    key: str = typer.Argument(..., help="Canonical key or citation of the judgment, e.g. INSC:2019:770."),
) -> None:
    """Is this judgment still good law? What every later judgment in the corpus did with it."""
    with get_session() as session:
        judgment = session.scalars(select(Judgment).where(Judgment.canonical_key == key)).first()
        if judgment is None:
            found = extract_citations(key)
            resolution = resolve_citation(session, found[0]) if found else None
            if resolution is None or not resolution.judgment_id:
                console.print(f"[red]{key}[/red] not in the knowledge base")
                raise typer.Exit(1)
            judgment = session.get(Judgment, resolution.judgment_id)

        report = citator.treatment_of(session, judgment.id)
        colour = "red" if report.is_doubtful else "green"
        console.print(f"[bold]{judgment.title[:80]}[/bold] [dim]{judgment.canonical_key}[/dim]")
        console.print(f"[{colour}]{report.status.replace('_', ' ')}[/{colour}]  ", end="")
        console.print(f"[dim]cited by {report.citing_count} of {report.corpus_size:,} judgments held[/dim]")
        if report.note:
            console.print(f"  {report.note}")

        for link in report.undermined_by:
            console.print(f"  [yellow]rests on[/yellow] {link}")

        if report.edges:
            table = Table(show_header=True, header_style="bold")
            for column in ("treatment", "citing judgment", "date", "bench", "para"):
                table.add_column(column)
            for edge in sorted(report.edges, key=lambda e: (not e.is_negative, e.citing_date or "")):
                label = edge.treatment.replace("_", " ")
                if edge.is_negative:
                    label = f"[red]{label}[/red]"
                if edge.downgraded_from:
                    label += f" [dim](claimed {edge.downgraded_from})[/dim]"
                table.add_row(
                    label,
                    edge.citing_title[:44],
                    edge.citing_date or "?",
                    str(edge.citing_bench or "?"),
                    edge.paragraph_label or "-",
                )
            console.print(table)


@app.command("argue")
def argue_command(
    file: str = typer.Argument(..., help="A file of propositions, one per line or paragraph."),
    checked: int = typer.Option(
        authority.DEFAULT_CHECKED, help="How many candidate authorities to verify per proposition."
    ),
    report: str | None = typer.Option(None, help="Also write the result to this file."),
) -> None:
    """Bind each proposition you intend to argue to an authority, or refuse to.

    The drafting half of the engine. For every proposition the corpus is searched for a judgment that
    says it, the candidates are put through the same verifier that checks a brief, and a gate decides
    whether any of them is fit to cite: the court's own words, the majority, still good law, and a
    quote that verifies word for word. A proposition that passes comes back with its pinpoint and its
    quote. One that does not comes back with what was considered and what is wrong with it.
    """
    init_db()
    propositions = [
        line.strip()
        for line in Path(file).read_text(encoding="utf-8").split("\n")
        if len(line.split()) >= 6
    ]
    if not propositions:
        console.print(f"[red]no propositions in {file}[/red]; one per line, six words or more")
        raise typer.Exit(1)

    model = build_structured(ScopeAssessment)
    if model is None:
        console.print(
            "[yellow]no language model configured[/yellow], so nothing can be *bound*: the gate can "
            "check voice, opinion and treatment, but not whether the paragraph says what you claim. "
            "Lines are offered to read, marked unchecked."
        )

    lines: list[str] = []
    with get_session() as session:
        if not search.index_exists(session):
            console.print("[dim]building the full-text index (first run)...[/dim]")
            search.build_index(session)
        for index, proposition in enumerate(propositions, start=1):
            console.print(f"[dim]{index}/{len(propositions)}[/dim]")
            binding = authority.bind_proposition(session, proposition, model, checked=checked)
            lines.extend(authority.render_binding(binding))
            lines.append("")

    console.print("\n".join(lines))
    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")
        console.print(f"[dim]written to {report}[/dim]")


@app.command("draft")
def draft_command(
    file: str = typer.Argument(..., help="A case plan. See `drafting/plan.py` for the format."),
    docx: str | None = typer.Option(None, help="Write the submission as a .docx here."),
    markdown: str | None = typer.Option(None, help="Write the submission as Markdown here."),
    checked: int = typer.Option(
        authority.DEFAULT_CHECKED, help="How many candidate authorities to verify per proposition."
    ),
    show: bool = typer.Option(True, help="Print the submission. --no-show for files only."),
    attack: bool = typer.Option(
        True, help="Add what the other side will say. --no-attack to leave it out."
    ),
    challenge: bool = typer.Option(
        False,
        "--challenge",
        help="Read every verified quote back against the claim. Off: it measures worse, see challenge.py.",
    ),
) -> None:
    """Assemble a written submission from a case plan, with every citation checked.

    The plan is the advocate's: the court, the parties, the issues, the propositions they intend to
    argue, and the prayer. What this adds is an authority behind each proposition or a mark saying
    there is none, a list of authorities built only from what was verified, and an appendix giving the
    paragraph and the words behind every citation in the document.

    A proposition with no authority is not dropped. It stays where it was written, marked, because a
    draft that quietly loses the sentences nothing could support reads as though everything in it is
    supported, and that is the document this engine exists to catch.
    """
    init_db()
    try:
        plan = read_plan(file)
    except PlanError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error
    except FileNotFoundError as error:
        console.print(f"[red]no such file: {file}[/red]")
        raise typer.Exit(1) from error

    model = build_structured(ScopeAssessment)
    if model is None:
        console.print(
            "[yellow]no language model configured[/yellow], so nothing can be bound: every "
            "proposition will come back marked, and the draft will be a list of what still needs an "
            "authority. That is a useful document, but it is not a submission."
        )

    # The second reading, off unless asked for. It can only lower support, so it cannot make the gate
    # less careful -- but measured on Gemini 2.5 Flash it tracked the prompt's framing rather than the
    # texts, so it is not on by default. `engine/challenge.py` carries the numbers.
    challenge_model = build_structured(ChallengeAssessment) if challenge and model else None
    if challenge and challenge_model is None and model is not None:
        console.print("[yellow]no model for the second reading[/yellow], so quotes go unchallenged")

    propositions = plan.propositions
    bindings = {}
    with get_session() as session:
        if not search.index_exists(session):
            console.print("[dim]building the full-text index (first run)...[/dim]")
            search.build_index(session)
        for index, proposition in enumerate(propositions, start=1):
            console.print(f"[dim]{index}/{len(propositions)} {proposition[:60]}[/dim]")
            bindings[proposition] = authority.bind_proposition(
                session, proposition, model, checked=checked, challenge_model=challenge_model
            )

    draft = assemble(plan, bindings)
    attacks = None
    if attack:
        with get_session() as session:
            attacks = attack_draft(session, draft)
    text = to_markdown(draft, attacks)
    if show:
        console.print(text)

    counts = draft.counts
    console.print(
        f"[bold]{counts['bound']} bound[/bold], {counts['narrowed']} narrowed, "
        f"{counts['unchecked']} unchecked, [red]{counts['refused']} without authority[/red]"
    )
    if draft.unsupported:
        console.print(
            f"[yellow]{len(draft.unsupported)} propositions carry no verified authority[/yellow] "
            "and are marked in the draft. Read them before filing."
        )
    if attacks:
        console.print(f"[yellow]{len(attacks)} things the other side will say[/yellow], in the appendix")

    if markdown:
        Path(markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(markdown).write_text(text, encoding="utf-8")
        console.print(f"[dim]written to {markdown}[/dim]")
    if docx:
        written = write_docx(draft, docx, attacks)
        console.print(f"[dim]written to {written}[/dim]")


@app.command("contrary")
def contrary_command(
    proposition: str = typer.Argument(..., help="The proposition you intend to argue."),
    top: int = typer.Option(5, help="How many leads to show."),
    against: str | None = typer.Option(
        None, "--against", help="The judgment you are relying on: its dissents, and it is left out."
    ),
    field_size: int = typer.Option(
        contrary.DEFAULT_FIELD, "--field", help="How many retrieved paragraphs to read for opposition."
    ),
    expand: bool = typer.Option(
        True, help="Also search for the proposition with its terms of art swapped for their opposites."
    ),
    check: bool = typer.Option(
        False, "--check", help="Have a model read each lead against the proposition. Off by default."
    ),
) -> None:
    """What the other side will cite: a judgment stating the opposite of what you argue.

    Every other check in this engine is about the authority you have. This one is about the ones you
    do not: it searches the corpus for a court, in its own voice, saying the other thing.

    What comes back is a lead. The engine has established that a court wrote a sentence on this
    subject with the opposite polarity — not that the sentence contradicts you rather than confining
    the rule to other facts, which is a reading. Read it before your opponent does.
    """
    init_db()
    with get_session() as session:
        if not search.index_exists(session):
            console.print("[red]no search index[/red]; run [bold]orderorder index[/bold] first")
            raise typer.Exit(1)

        exclude: tuple[str, ...] = ()
        relied_on = None
        if against:
            relied_on = session.scalars(
                select(Judgment).where(Judgment.canonical_key == against)
            ).first()
            if relied_on is None:
                console.print(f"[red]{against} is not in the corpus[/red]")
                raise typer.Exit(1)
            exclude = (relied_on.id,)

        report = contrary.find_contrary(
            session, proposition, exclude=exclude, top=top, field_size=field_size, expand=expand
        )
        dissents = contrary.dissents_in(session, relied_on.id, proposition) if relied_on else []

        if check:
            model = build_structured(OppositionAssessment)
            if model is None:
                console.print(
                    "[yellow]no language model configured[/yellow], so these stay leads: a court "
                    "wrote them on this subject with the opposite polarity, and nothing has read "
                    "them against your proposition."
                )
            else:
                contrary.read_leads(proposition, report, model)

        for lead in dissents:
            console.print(
                f"\n[bold yellow]the dissent in the case you rely on[/bold yellow]\n"
                f"   [cyan]{lead.authority.canonical_key}, para {lead.authority.paragraph_label}[/cyan]  "
                f"[dim]{lead.opposition.cue}[/dim]"
            )
            console.print(f'   [yellow]"{lead.sentence[:300]}"[/yellow]')

        for rank, lead in enumerate(report.leads, start=1):
            authority_found = lead.authority
            console.print(
                f"\n[bold]{rank}. {authority_found.title[:70]}[/bold]\n"
                f"   [cyan]{authority_found.pinpoint}[/cyan]  "
                f"[dim]{authority_found.decided_on or '?'} · bench {authority_found.bench_strength or '?'}"
                f" · {lead.opposition.describe()}[/dim]"
            )
            console.print(f'   [red]"{lead.sentence[:300]}"[/red]')
            if lead.reading is not None:
                colour = "red" if lead.is_confirmed else "yellow"
                console.print(f"   [{colour}]read[/{colour}]: {lead.reading.describe()}")

        console.print(f"\n[dim]{report.describe()}.[/dim]")
        if report.found:
            console.print(
                "[dim]Each is a passage to read, not a holding that you are wrong: a court can state "
                "a rule narrowly without contradicting a wider one.[/dim]"
            )


@app.command("find")
def find_command(
    proposition: str = typer.Argument(..., help="The proposition you want an authority for."),
    top: int = typer.Option(5, help="How many authorities to show."),
    candidates: int = typer.Option(
        search.DEFAULT_CANDIDATES, help="Paragraphs retrieved before authority is weighed."
    ),
    any_voice: bool = typer.Option(
        False, "--any-voice", help="Keep passages that are not the court speaking. Off by default."
    ),
    check: bool = typer.Option(True, help="Run the verifier over each authority found."),
    dense: bool = typer.Option(
        False, "--dense", help="Fuse the vector ranking in too, if a store has been built."
    ),
    role: list[str] = typer.Option(
        None,
        "--role",
        help="Restrict to paragraphs labelled with this rhetorical role (ingest mark-roles). "
        "Repeatable: --role ratio --role analysis.",
    ),
    role_boost: bool = typer.Option(
        False,
        "--role-boost",
        help="Lift labelled holdings above narration at near-ties. Measured: +6 fragment recall, "
        "-6 paraphrase -- on when you are stating the law, off when finding a passage.",
    ),
) -> None:
    """Search every judgment for an authority backing a proposition, and name the line.

    The opposite direction from `verify`: no citation is given, so the corpus is searched for one.
    """
    init_db()
    with get_session() as session:
        if not search.index_exists(session):
            console.print("[dim]building the full-text index (first run)...[/dim]")
            search.build_index(session)

        authorities = search.find_authorities(
            session, proposition, top=top, candidates=candidates, court_voice_only=not any_voice,
            dense=dense, roles=list(role) or None, role_boost=role_boost,
        )
        if not authorities:
            console.print(
                "[yellow]nothing found[/yellow]. The corpus holds text for "
                f"{session.scalar(select(func.count()).select_from(JudgmentTextVersion)) or 0} judgments; "
                "run [bold]orderorder ingest bulk-text[/bold] for more, or rephrase the proposition."
            )
            raise typer.Exit(1)

        model = build_structured(ScopeAssessment) if check else None
        if check and model is None:
            console.print(
                "[yellow]no language model configured[/yellow], so these are candidates the words match, "
                "not authorities confirmed to support you. Add a provider key to have each one checked."
            )

        for rank, authority in enumerate(authorities, start=1):
            console.print(
                f"\n[bold]{rank}. {authority.title[:70]}[/bold]\n"
                f"   [cyan]{authority.pinpoint}[/cyan]  "
                f"[dim]{authority.decided_on or '?'} · bench {authority.bench_strength or '?'} · "
                f"score {authority.score:.2f}[/dim]"
            )
            if authority.treatment is not None and authority.treatment.is_doubtful:
                colour = "yellow" if authority.treatment.is_undermined else "red"
                console.print(
                    f"   [{colour}]{authority.treatment.status.replace('_', ' ').upper()}[/{colour}] "
                    f"{authority.treatment.note}"
                )
            if authority.line:
                console.print(f'   [green]"{authority.line[:300]}"[/green]')
            if authority.voice and not authority.voice.is_the_court:
                console.print(f"   [yellow]voice[/yellow]: {authority.voice.voice}")

            if model is not None:
                verdict = _check_authority(session, authority, proposition, model)
                colour = {"full": "green", "partial": "yellow"}.get(verdict.support, "red")
                console.print(f"   [{colour}]checked[/{colour}]: {verdict.support}", end="")
                if verdict.scope and verdict.scope.gap:
                    console.print(f" — {verdict.scope.gap}")
                else:
                    console.print()

        console.print(f"\n[dim]{len(authorities)} authorities from the local corpus.[/dim]")


def _check_authority(session, authority, proposition: str, model):
    """Run the verifier over one retrieved authority: does that paragraph really support the claim?"""
    from orderorder.engine.locator import locate as locate_in
    from orderorder.engine.scope import assess_scope
    from orderorder.engine.verdict import build_verdict
    from orderorder.resolver import Resolution

    paragraphs = load_paragraphs(session, authority.judgment_id)
    label = authority.paragraph_label
    location = locate_in(paragraphs, proposition, claimed_pinpoint=label, top_k=6)
    scope = assess_scope(proposition, location.candidates, model)
    return build_verdict(
        authority.citation or authority.canonical_key,
        proposition,
        Resolution(
            status="found",
            method="search",
            judgment_id=authority.judgment_id,
            canonical_key=authority.canonical_key,
            score=100.0,
        ),
        judgment_title=authority.title,
        pinpoint=location.pinpoint,
        scope=scope,
        voice=authority.voice,
        claimed_pinpoint=label,
    )


@app.command("verify")
def verify_command(
    text: str = typer.Argument(None, help="Text containing citations. Omit to read from --file."),
    file: str | None = typer.Option(None, "--file", "-f", help="Read the brief from this file."),
    top: int = typer.Option(6, help="Candidate paragraphs considered per citation."),
    show_quote: bool = typer.Option(True, help="Print the verified quote for supported claims."),
    facts: str | None = typer.Option(
        None, "--facts", help="File of the present matter's facts, to test each authority against them."
    ),
    memo: bool = typer.Option(
        False,
        "--memo",
        help="Write what opposing counsel would say about each citation, and how to answer it.",
    ),
    report: str | None = typer.Option(
        None, "--report", help="Write a verification report covering every citation to this file."
    ),
    annotate: str | None = typer.Option(
        None, "--annotate", help="Write a copy of the brief with a flag beside every citation."
    ),
    no_model: bool = typer.Option(
        False,
        "--no-model",
        help="Run only the checks that need no model, even where one is configured.",
    ),
) -> None:
    """Verify every citation in a passage: does the case exist, which paragraph, and does it support the claim."""
    if file:
        text = Path(file).read_text(encoding="utf-8")
    if not text or not text.strip():
        console.print("[red]give some text, or --file[/red]")
        raise typer.Exit(1)

    # One model per question, because each is bound to its own output schema. Whose words a passage
    # carries is decided from the judgment's structure, so that check survives having no key at all.
    matter_facts = Path(facts).read_text(encoding="utf-8") if facts else ""
    # Eight of the twelve failure modes need no model, and being able to say so is only worth
    # anything if it can be demonstrated on a machine that does have one configured.
    model = None if no_model else build_structured(ScopeAssessment)
    if model is None:
        console.print(
            "[yellow]no language model configured[/yellow] so existence, pinpoint, voice and opinion "
            "are checked but extent of support is not. Set a provider key; see .env.example."
        )

    with get_session() as session:
        verdicts = verify_text(
            session,
            text,
            model,
            top_k=top,
            voice_model=None if no_model else build_structured(VoiceAssessment),
            weight_model=None if no_model else build_structured(WeightAssessment),
            facts_model=(
                None if no_model or not matter_facts else build_structured(ApplicabilityAssessment)
            ),
            matter_facts=matter_facts,
        )

    if not verdicts:
        console.print("[yellow]no citations found[/yellow]")
        raise typer.Exit(1)

    board = Table(show_header=True, header_style="bold", title="Verdict board")
    board.add_column("grade", justify="center")
    board.add_column("citation")
    board.add_column("case")
    board.add_column("support")
    board.add_column("voice")
    board.add_column("law")
    board.add_column("findings")
    colours = {"A": "green", "B": "green", "C": "yellow", "D": "yellow", "E": "red", "F": "red"}
    for v in verdicts:
        colour = colours.get(v.grade, "white")
        board.add_row(
            f"[{colour}]{v.grade}[/{colour}]",
            v.citation_raw,
            (v.judgment_title or "-")[:40],
            v.support,
            _voice_badge(v),
            _law_badge(v),
            str(len(v.findings)) if v.findings else "-",
        )
    console.print(board)

    for v in verdicts:
        if not v.findings and not v.needs_review:
            continue
        console.print(f"\n[bold]{v.citation_raw}[/bold] [dim]{(v.judgment_title or '')[:60]}[/dim]")
        console.print(f"  claim: [italic]{_shorten(v.proposition, 150)}[/italic]")
        for finding in v.findings:
            console.print(f"  [red]{finding}[/red]")
        if v.needs_review:
            console.print(f"  [yellow]needs review[/yellow]: {v.review_reason}")
        if v.applicability is not None and v.applicability.is_problem:
            console.print(f"  [magenta]distinguishable[/magenta]: {v.applicability.reason}")
        if v.hierarchy is not None and v.hierarchy.is_problem:
            console.print(f"  [red]wrong court[/red]: {v.hierarchy.note}")
        if v.treatment is not None and v.treatment.is_doubtful:
            console.print(f"  [red]dead law[/red]: {v.treatment.note}")
        if v.voice is not None and (v.voice.is_problem or v.voice.is_dissent):
            where = v.paragraph_label or v.claimed_pinpoint or "?"
            console.print(f"  [magenta]voice[/magenta] (para {where}): {v.voice.reason}")
        if v.weight is not None and v.weight.is_obiter:
            console.print(f"  [magenta]weight[/magenta]: {v.weight.reason}")
        if show_quote and v.quote_verified and v.quote:
            console.print(
                f'  [green]verified quote[/green] (para {v.paragraph_label}): "{_shorten(v.quote, 160)}"'
            )
        if v.scope and v.scope.narrowed_proposition:
            console.print(f"  [cyan]supported instead[/cyan]: {v.scope.narrowed_proposition}")

    if memo:
        # Weakest first: the memo is read to decide what to drop, and that decision starts at the
        # bottom of the board.
        console.print("\n[bold]Opposing counsel's memo[/bold]")
        for verdict in sorted(verdicts, key=lambda v: -GRADES.index(v.grade)):
            console.print("\n".join(render_memo(write_memo(verdict))))

    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        Path(report).write_text(
            verification_report(verdicts, source=file or "pasted text"), encoding="utf-8"
        )
        console.print(f"[dim]report written to {report}[/dim]")
    if annotate:
        Path(annotate).parent.mkdir(parents=True, exist_ok=True)
        Path(annotate).write_text(annotate_brief(text, verdicts), encoding="utf-8")
        console.print(f"[dim]annotated brief written to {annotate}[/dim]")

    graded = sum(1 for v in verdicts if v.grade in "DEF")
    console.print(
        f"\n{len(verdicts)} citations, {graded} graded D or worse, "
        f"{sum(1 for v in verdicts if v.needs_review)} needing review"
    )


@cite_app.command("parse")
def cite_parse(text: str = typer.Argument(..., help="Text to scan for citations.")) -> None:
    """Show every citation the grammar finds, with its normalised key and pinpoint."""
    found = extract_citations(text)
    if not found:
        console.print("[yellow]no citations found[/yellow]")
        raise typer.Exit(1)
    table = Table(show_header=True, header_style="bold")
    for column in ("raw", "reporter", "normalized", "pinpoint", "parties"):
        table.add_column(column)
    for citation in found:
        table.add_row(
            citation.raw,
            citation.reporter,
            citation.normalized,
            citation.pinpoint.label if citation.pinpoint else "-",
            citation.party_names or "-",
        )
    console.print(table)


@app.command("resolve")
def resolve_command(text: str = typer.Argument(..., help="A citation, optionally with party names.")) -> None:
    """Resolve every citation in the text against the knowledge base."""
    found = extract_citations(text)
    if not found:
        console.print("[yellow]no citations found[/yellow]")
        raise typer.Exit(1)
    with get_session() as session:
        for citation in found:
            result = resolve_citation(session, citation)
            colour = {"found": "green", "ambiguous": "yellow"}.get(result.status, "red")
            console.print(
                f"[{colour}]{result.status}[/{colour}] {citation.raw} "
                f"-> {result.canonical_key or '-'} via {result.method} ({result.score:.0f})"
            )
            if result.note:
                console.print(f"  [dim]{result.note}[/dim]")
            for candidate in result.candidates[:3]:
                console.print(
                    f"  [dim]{candidate.score:5.1f} {candidate.canonical_key} {candidate.title[:70]}[/dim]"
                )


@app.command()
def stats() -> None:
    """Row counts for the knowledge base."""
    with get_session() as session:
        table = Table(show_header=True, header_style="bold")
        table.add_column("table")
        table.add_column("rows", justify="right")
        for name, model in [
            ("judgment", Judgment),
            ("citation_alias", CitationAlias),
            ("judgment_text_version", JudgmentTextVersion),
            ("paragraph", Paragraph),
        ]:
            count = session.scalar(select(func.count()).select_from(model)) or 0
            table.add_row(name, f"{count:,}")
        console.print(table)
        earliest, latest = session.execute(
            select(func.min(Judgment.decided_on), func.max(Judgment.decided_on))
        ).first() or (None, None)
        if earliest:
            console.print(f"date range: {earliest} to {latest}")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
