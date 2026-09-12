"""The engine behind a browser.

`docs/PRD.md` A14: a verdict board, an annotated brief, a judgment viewer with the paragraph
highlighted. All three exist as text already; what a page adds is that they can be read *together*.
An advocate revising a memorial wants to click a flag, see the paragraph the engine actually read, and
see the sentence highlighted inside it — three things that are three commands and a lot of scrolling
at a terminal.

**No Node.** The architecture names Next.js, and this serves one static HTML file instead. The reason
is not laziness about the framework: the engine's output is a list of verdicts and a document to mark
up, which is a page rather than an application, and a build toolchain would put an npm install and a
bundler between `uv run orderorder serve` and a working demo — on a machine whose system drive has no
room for one. If the drafting workspace grows into something with real client state, that is the point
to reconsider, and nothing here would have to be thrown away: the API is the same either way.

Every route is read-only against the knowledge base. Nothing this serves can change a judgment.

The app is handed the way it opens a session rather than reaching for the global one. That is what
lets a test point it at a database of its own with certainty — the module-level caches behind
`get_session` can be left stale by anything that reloads the config module, and a test that
silently talks to the real nine-thousand-judgment corpus is worse than one that fails.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from orderorder import __version__, logs
from orderorder.agent import NoModelConfigured, build_assistant
from orderorder.db.models import Judgment, JudgmentTextVersion
from orderorder.db.session import get_session
from orderorder.drafting.assemble import assemble
from orderorder.drafting.attack import attack_draft
from orderorder.drafting.plan import PlanError, parse_plan
from orderorder.drafting.render import to_markdown
from orderorder.drafting.word import write_docx
from orderorder.engine import authority as authority_gate
from orderorder.engine import search
from orderorder.engine.graph import verify_text
from orderorder.engine.memo import render_memo, write_memo
from orderorder.engine.providers import build_structured
from orderorder.engine.report import annotate_brief, verification_report
from orderorder.engine.schemas import (
    ApplicabilityAssessment,
    ScopeAssessment,
    VoiceAssessment,
    WeightAssessment,
)
from orderorder.ingest.brief import read_brief
from orderorder.ingest.store import load_paragraphs
from orderorder.web import limits
from orderorder.web.auth import TOKEN_ENV, token_required
from orderorder.web.jobs import (
    DraftJob,
    Job,
    JobStore,
    Watched,
    as_json,
    binding_json,
    watch,
    watch_draft,
)

log = logs.get_logger(__name__)

STATIC = Path(__file__).parent / "static"

# Sent on every response, including the refusals, which is why they go through one helper.
#
# The page is one origin serving itself. It loads no font, no script and no stylesheet from anywhere
# else, and it calls no API but its own, so the policy can say exactly that rather than the usual
# hedge. `script-src 'self'` with no `'unsafe-inline'` is the load-bearing line and the reason the
# page's script and style live in their own files: an injected `<script>` in a citation, a party name
# or a paragraph of a judgment does not execute, whatever escaping elsewhere may have missed.
#
# `frame-ancestors 'none'` and `X-Frame-Options` say the same thing twice on purpose; the header is
# what an older browser understands. HSTS is not here: this server is meant to sit behind a proxy that
# terminates TLS, and a service that cannot see whether it is on HTTPS should not be the one asserting
# that it always will be. `docs/DEPLOYMENT.md` §2.2 says where it belongs.
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self'; "
        "img-src 'self' data:; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "object-src 'none'; "
        "base-uri 'none'; "
        "form-action 'self'; "
        "frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=(), payment=(), usb=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}


def _secured(headers: dict | None = None) -> dict:
    """Response headers with the security set applied. A refusal gets them too."""
    merged = dict(SECURITY_HEADERS)
    merged.update(headers or {})
    return merged


# How long a case plan may be. A plan is issues and sentences, not a bundle; past this something
# other than a plan has been pasted in.
MAX_PLAN_CHARS = 60_000

# How long a brief may be. A memorial is tens of kilobytes; anything past this is a book, and checking
# it would tie the single worker up for an hour with no way to say so.
MAX_BRIEF_CHARS = 400_000
# A question may carry a passage for the agent to check, so this is not a tweet-sized limit; but it is
# far short of MAX_BRIEF_CHARS, because a question goes into a model's context whole and a 400,000
# character one would be refused by the provider after the request had been accepted here. A brief that
# long belongs in /api/verify, which streams it a citation at a time.
MAX_QUESTION_CHARS = 60_000
# And how large a file. A memorial is under a megabyte; a bundle of annexures is not a brief.
MAX_UPLOAD_BYTES = 25_000_000
# How much is read at a time while enforcing that limit.
UPLOAD_CHUNK = 1 << 20
# What a brief may arrive as. `read_brief` decides for real by parsing; this refuses the rest before
# anything is read into memory.
ACCEPTED_UPLOADS = (".pdf", ".docx", ".txt", ".md")
# Named formats that cannot be read but that a person will reasonably try, so they are let past the
# gate for `read_brief` to turn down in a sentence that says what to do instead. Refusing `.doc` with
# a bare "unsupported media type" would be correct and useless: the advocate has the file, and what
# they need is "save it as .docx", not a status code.
EXPLAINED_UPLOADS = (".doc",)

# The faces the page asks for, as a set the font route matches against rather than a path it joins.
FONT_FILES = frozenset(
    {
        "geist-latin.woff2",
        "geist-latin-ext.woff2",
        "geist-mono-latin.woff2",
        "geist-mono-latin-ext.woff2",
        "playfair-latin.woff2",
        "playfair-italic-latin.woff2",
    }
)


async def _read_at_most(file: UploadFile, limit: int) -> bytes | None:
    """The whole file, or `None` if it is larger than `limit`.

    `await file.read()` reads all of it and *then* the caller checks the size, which is a check that
    has already lost: the memory is spent by the time it runs, and the number that decides how much is
    a stranger's upload. `Content-Length` cannot be trusted to fix it either -- it is a header, and a
    chunked request need not send one. So the read itself is bounded, one megabyte at a time, and it
    stops the moment the total goes past the limit rather than after.
    """
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(UPLOAD_CHUNK):
        total += len(chunk)
        if total > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


class DraftRequest(BaseModel):
    plan: str = Field(description="A case plan. See `drafting/plan.py` for the format.")
    use_model: bool = Field(
        default=True,
        description="Run the checks that need a language model. Without one nothing can be bound.",
    )


class AgentRequest(BaseModel):
    question: str = Field(description="A question in plain words, with any passage it refers to.")


class VerifyRequest(BaseModel):
    text: str = Field(description="The brief, memorial or passage to check.")
    facts: str = Field(default="", description="The facts of the present matter, if there are any.")
    use_model: bool = Field(
        default=True,
        description="Run the checks that need a language model. Off runs the eight that do not.",
    )


def create_app(
    *, store: JobStore | None = None, session_factory=get_session, token: str | None = None
) -> FastAPI:
    # Here as well as in `serve`, because uvicorn can be pointed at the module-level app directly and
    # a deployment that did that would otherwise run silent. Calling it twice changes nothing.
    logs.configure()
    app = FastAPI(title="OrderOrder", version=__version__, docs_url="/api/docs")

    limiter = limits.RateLimiter()

    # One place, checked before anything else runs. A token configured per-route is a token somebody
    # forgets on the route added next week, and the route added next week is the upload endpoint.
    @app.middleware("http")
    async def _guard(request, call_next):
        from fastapi.responses import JSONResponse
        from starlette.exceptions import HTTPException as StarletteHTTPException

        def refuse(status: int, detail: str, headers: dict | None = None) -> JSONResponse:
            return JSONResponse({"detail": detail}, status_code=status, headers=_secured(headers))

        who = limits.client_of(request)

        # Expired windows go here, before anything can refuse this request. The limiter's dictionary
        # is keyed by client address, so without a sweep it is a leak an attacker grows on purpose --
        # and a flood of wrong tokens, which is the cheapest way to grow it, never reaches a route.
        limiter.maybe_prune()

        # A body large enough to hurt is refused before it is read. `Content-Length` is a claim rather
        # than a fact, so this is the cheap half; `_read_at_most` is what actually bounds an upload.
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > limits.MAX_BODY_BYTES:
            return refuse(413, f"a request body may be up to {limits.MAX_BODY_BYTES // 1_000_000} MB")

        try:
            token_required(request, token)
        except StarletteHTTPException as refused:
            # Count the failure, then refuse. Counting first means a client that is already over the
            # limit is told to wait rather than told its token was wrong, which is one less signal.
            wait = limiter.allow(who, "auth", limit=limits.AUTH_FAILURES)
            if wait is not None:
                return refuse(429, "too many failed attempts", {"Retry-After": str(int(wait) + 1)})
            return refuse(refused.status_code, refused.detail, dict(refused.headers or {}))

        bucket = "write" if request.method in limits.WRITE_METHODS else "read"
        ceiling = limits.WRITE_REQUESTS if bucket == "write" else None
        wait = limiter.allow(who, bucket, limit=ceiling)
        if wait is not None:
            return refuse(
                429,
                "too many requests; this server checks one brief at a time",
                {"Retry-After": str(int(wait) + 1)},
            )

        response = await call_next(request)
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        return response
    # `is not None`, not `or`: a JobStore defines __len__, so an empty one is falsy and `or` would
    # quietly hand back a different store than the caller passed in.
    jobs = store if store is not None else JobStore()
    open_session = session_factory

    @app.get("/api/health")
    def health() -> dict:
        """What the corpus holds and whether a model can be reached, for the page's status line."""
        with open_session() as session:
            judgments = session.scalar(select(Judgment.id).limit(1))
            held = session.query(JudgmentTextVersion).count()
        return {
            "version": __version__,
            "corpus_ready": judgments is not None,
            "judgments_with_text": held,
            "model_configured": build_structured(ScopeAssessment) is not None,
            "agent": _agent_status(),
        }

    @app.post("/api/verify")
    def start_verification(request: VerifyRequest) -> dict:
        """Take a brief and start checking it. Returns at once with a job to watch."""
        text = request.text.strip()
        if not text:
            raise HTTPException(400, "there is nothing to check")
        if len(text) > MAX_BRIEF_CHARS:
            raise HTTPException(413, f"a brief may be up to {MAX_BRIEF_CHARS:,} characters")

        job = jobs.create(text, source="pasted text")
        job.model_configured = request.use_model and build_structured(ScopeAssessment) is not None
        threading.Thread(target=_run, args=(job, request, open_session), daemon=True).start()
        return {"job": job.id, "model_configured": job.model_configured}

    @app.post("/api/upload")
    async def upload(file: UploadFile) -> dict:
        """Read a brief out of a PDF or DOCX and hand back the text, without checking anything yet.

        The text is shown before it is checked, on purpose. A verdict on a document the reader has not
        seen is a verdict neither of you can point at, and a PDF that came out mangled should be
        obvious in a second rather than discovered through nine inexplicable phantom citations.
        """
        name = file.filename or ""
        if not name.lower().endswith(ACCEPTED_UPLOADS + EXPLAINED_UPLOADS):
            # Named before it is read. Deciding by extension is not a security control on its own --
            # the parsers below are what actually decide -- but it refuses the obviously wrong file
            # without first spending memory on it, and it gives the reader a sentence they can act on.
            raise HTTPException(
                415, f"a brief has to be one of {', '.join(ACCEPTED_UPLOADS)}; that is {name or 'unnamed'}"
            )

        data = await _read_at_most(file, MAX_UPLOAD_BYTES)
        if data is None:
            raise HTTPException(413, f"a file may be up to {MAX_UPLOAD_BYTES // 1_000_000} MB")
        try:
            brief = read_brief(data, file.filename or "")
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - a broken file must not read as an empty brief
            raise HTTPException(400, f"that file could not be read: {type(exc).__name__}") from exc

        if not brief.text.strip():
            raise HTTPException(
                422,
                brief.note
                or "no text came out of that file; if it is a scan it needs OCR before it can be checked",
            )
        return {
            "text": brief.text,
            "kind": brief.kind,
            "pages": brief.pages,
            "note": brief.note,
            "filename": file.filename,
        }

    @app.get("/api/jobs/{job_id}")
    def read_job(job_id: str) -> dict:
        job = _find(jobs, job_id)
        return {
            "job": job.id,
            "done": job.done,
            "total": job.total,
            "finished": job.finished,
            "error": job.error,
            "model_configured": job.model_configured,
            "verdicts": [as_json(v) for v in job.verdicts],
        }

    @app.get("/api/jobs/{job_id}/events")
    def stream_job(job_id: str) -> StreamingResponse:
        """Server-sent events: one per verdict as it lands, then done.

        The board fills in the order the engine works, so a phantom citation — settled against the
        alias table in milliseconds — is on screen while the ones that need a model are still running.
        """
        job = _find(jobs, job_id)

        def events() -> Iterator[str]:
            for name, payload in watch(job):
                yield f"event: {name}\ndata: {json.dumps(payload)}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/jobs/{job_id}/annotated", response_class=PlainTextResponse)
    def annotated(job_id: str) -> str:
        job = _find(jobs, job_id)
        return annotate_brief(job.text, job.verdicts)

    @app.get("/api/jobs/{job_id}/report", response_class=PlainTextResponse)
    def report(job_id: str) -> str:
        job = _find(jobs, job_id)
        return verification_report(job.verdicts, source=job.source)

    @app.get("/api/jobs/{job_id}/memo/{index}", response_class=PlainTextResponse)
    def memo(job_id: str, index: int) -> str:
        job = _find(jobs, job_id)
        if not 0 <= index < len(job.verdicts):
            raise HTTPException(404, "no such citation in this job")
        return "\n".join(render_memo(write_memo(job.verdicts[index])))

    @app.post("/api/plan")
    def check_plan(request: DraftRequest) -> dict:
        """Read a plan and hand back what it says, without binding anything.

        The page shows the parsed issues and propositions before it runs, for the same reason the
        brief is shown before it is checked: an advocate should see what the tool thinks they wrote
        while a typo is still a typo, rather than after four minutes of model calls.
        """
        try:
            plan = parse_plan(request.plan)
        except PlanError as error:
            raise HTTPException(400, str(error)) from error
        return {
            "court": plan.court,
            "cause": plan.cause,
            "parties": plan.parties,
            "appearing_for": plan.appearing_for,
            "dates": plan.dates,
            "issues": [{"title": i.title, "propositions": i.propositions} for i in plan.issues],
            "prayer": plan.prayer,
            "propositions": len(plan.propositions),
        }

    @app.post("/api/draft")
    def start_draft(request: DraftRequest) -> dict:
        """Take a case plan and start binding it. Returns at once with a job to watch."""
        if len(request.plan) > MAX_PLAN_CHARS:
            raise HTTPException(413, f"a plan may be up to {MAX_PLAN_CHARS:,} characters")
        try:
            plan = parse_plan(request.plan)
        except PlanError as error:
            raise HTTPException(400, str(error)) from error

        job = jobs.create_draft(plan, source="pasted plan")
        job.total = len(plan.propositions)
        job.model_configured = request.use_model and build_structured(ScopeAssessment) is not None
        threading.Thread(target=_draft, args=(job, open_session), daemon=True).start()
        return {"job": job.id, "model_configured": job.model_configured, "total": job.total}

    @app.get("/api/draft/{job_id}")
    def read_draft(job_id: str) -> dict:
        job = _find_draft(jobs, job_id)
        return {
            "job": job.id,
            "done": job.done,
            "total": job.total,
            "finished": job.finished,
            "error": job.error,
            "model_configured": job.model_configured,
            "bindings": [binding_json(p, job.bindings[p]) for p in job.order],
            "attacks": [
                {"kind": a.kind, "says": a.says, "fix": a.fix, "citation": a.citation}
                for a in job.attacks
            ],
        }

    @app.get("/api/draft/{job_id}/events")
    def stream_draft(job_id: str) -> StreamingResponse:
        job = _find_draft(jobs, job_id)

        def events() -> Iterator[str]:
            for name, payload in watch_draft(job):
                yield f"event: {name}\ndata: {json.dumps(payload)}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/draft/{job_id}/document", response_class=PlainTextResponse)
    def draft_document(job_id: str) -> str:
        job = _find_draft(jobs, job_id)
        if job.document is None:
            raise HTTPException(409, "the draft is still being assembled")
        return job.document

    @app.get("/api/draft/{job_id}/document.docx")
    def draft_docx(job_id: str) -> FileResponse:
        """The submission as a Word file, written to a temporary path and handed straight back."""
        job = _find_draft(jobs, job_id)
        if job.document is None:
            raise HTTPException(409, "the draft is still being assembled")
        draft = assemble(job.plan, job.bindings)
        path = Path(tempfile.gettempdir()) / f"orderorder-{job.id}.docx"
        write_docx(draft, path, job.attacks)
        return FileResponse(
            path,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename="written-submissions.docx",
        )

    @app.get("/api/judgment/{key}")
    def judgment(key: str, highlight: str | None = None) -> dict:
        """A judgment's paragraphs, so the page can show the one the verdict rests on.

        `highlight` is the verified quote. It is found here rather than in the browser because the same
        normalisation that verified it has to find it: matching raw text in JavaScript would fail on
        exactly the curly quotes and soft hyphens the verifier exists to survive, and would then show
        nothing highlighted on a citation that is perfectly sound.

        The paragraph comes back already cut into the part before the quote, the quote, and the part
        after, rather than as a body and a pair of offsets. Python counts characters and JavaScript
        counts UTF-16 units, so the two agree only while nothing in the corpus lies outside the basic
        plane — true of these 409,499 paragraphs today and enforced by nothing. Three strings cannot
        drift.
        """
        from orderorder.engine.quotes import find_quote

        with open_session() as session:
            record = session.scalars(select(Judgment).where(Judgment.canonical_key == key)).first()
            if record is None:
                raise HTTPException(404, f"no judgment with key {key}")
            paragraphs = load_paragraphs(session, record.id)
            citation = _preferred_citation(session, record.id)

        out = []
        for paragraph in paragraphs:
            found = find_quote(highlight, paragraph.body) if highlight else None
            body, quoted, rest = paragraph.body, "", ""
            if found is not None and found.found:
                body = paragraph.body[: found.char_start]
                quoted = paragraph.body[found.char_start : found.char_end]
                rest = paragraph.body[found.char_end :]
            out.append(
                {
                    "seq": paragraph.seq,
                    "label": paragraph.printed_label,
                    "body": body,
                    "quoted": quoted,
                    "rest": rest,
                    "opinion": paragraph.opinion_kind,
                    "author": paragraph.opinion_author,
                }
            )
        return {
            "key": record.canonical_key,
            "title": record.title,
            "court": record.court,
            "decided_on": record.decided_on.isoformat() if record.decided_on else None,
            "bench_strength": record.bench_strength,
            "citation": citation,
            "paragraphs": out,
        }

    @app.get("/api/search")
    def find(q: str, top: int = 5) -> dict:
        """The other direction: a proposition in, judgments and the line out."""
        if not q.strip():
            raise HTTPException(400, "there is nothing to search for")
        with open_session() as session:
            if not search.index_exists(session):
                raise HTTPException(409, "the full-text index has not been built; run `orderorder index`")
            found = search.find_authorities(session, q, top=top)
            return {
                "query": q,
                "authorities": [
                    {
                        "key": a.canonical_key,
                        "title": a.title,
                        "citation": a.citation,
                        "pinpoint": a.pinpoint,
                        "paragraph": a.paragraph_label,
                        "decided_on": a.decided_on,
                        "bench_strength": a.bench_strength,
                        "score": a.score,
                        "line": a.line,
                        "voice": a.voice.voice if a.voice else None,
                        "doubtful": bool(a.treatment and a.treatment.is_doubtful),
                        "treatment_note": a.treatment.note if a.treatment else None,
                    }
                    for a in found
                ],
            }

    @app.post("/api/agent")
    def ask_agent(request: AgentRequest) -> dict:
        """A question in plain words. The agent picks which checks to run and reports what they said.

        Deliberately not `async def`. Every tool under `orderorder.agent.tools` does blocking database
        work and one of them runs the whole verification graph; awaiting that on the event loop would
        stall every other request this process is serving, including the health check a load balancer
        is using to decide whether it is still alive. A plain `def` goes to FastAPI's thread pool,
        where blocking is what the thread is for.

        A fresh agent per request, so that two people's questions cannot land in one conversation. The
        cost is that it remembers nothing between questions and a follow-up has to restate what it
        refers to. Continuity is a session manager's job; Strands has one, and docs/ROADMAP.md is
        where it belongs rather than here.

        `tools_used` comes back with every answer because it is the audit trail. An answer about
        subsequent history that never called `check_treatment` was answered from the model's memory --
        which the system prompt forbids -- and a caller should be able to see that without being given
        the server's logs.
        """
        question = request.question.strip()
        if not question:
            raise HTTPException(400, "there is no question")
        if len(question) > MAX_QUESTION_CHARS:
            raise HTTPException(413, f"a question may be up to {MAX_QUESTION_CHARS:,} characters")

        try:
            assistant = build_assistant(session_factory=open_session)
        except NoModelConfigured as exc:
            # 503 rather than 500: nothing is broken, something is unconfigured, and the distinction is
            # the difference between reading a stack trace and setting a key.
            raise HTTPException(503, str(exc)) from exc

        try:
            answer = assistant.ask(question)
        except Exception as exc:  # noqa: BLE001 - the caller needs an answer, whatever went wrong
            log.warning("agent question failed: %s", logs.reason(exc))
            raise HTTPException(502, f"{type(exc).__name__}: {exc}") from exc

        return {
            "question": question,
            "answer": answer,
            "model": assistant.model.as_json(),
            "tools_used": assistant.tools_used(),
        }

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    # Named one by one rather than mounted as a directory. A `StaticFiles` mount serves whatever is
    # under the directory, which is a decision made once and then inherited by every file anybody
    # drops there later; three explicit routes cannot serve a fourth file by accident and have no path
    # for a caller to traverse. There are only ever going to be three.
    @app.get("/app.css")
    def stylesheet() -> FileResponse:
        return FileResponse(STATIC / "app.css", media_type="text/css")

    @app.get("/app.js")
    def script() -> FileResponse:
        return FileResponse(STATIC / "app.js", media_type="text/javascript")

    @app.get("/fonts/{name}")
    def font(name: str) -> FileResponse:
        """The four self-hosted Geist faces, served by name from a fixed list.

        Named rather than mounted, and matched against a set rather than joined onto a path: `name`
        arrives from the URL, and `STATIC / name` with a `..` in it is the oldest file-serving bug
        there is. A membership test cannot traverse anywhere.

        Self-hosted because the Content-Security-Policy is `font-src 'self'`. Loading these from
        Google would mean widening the policy to a third-party origin, and a typeface is not worth
        that; 82 KB in the repository is the cheaper trade.
        """
        if name not in FONT_FILES:
            raise HTTPException(404, "no such font")
        return FileResponse(STATIC / "fonts" / name, media_type="font/woff2")

    return app


def _find(jobs: JobStore, job_id: str) -> Job:
    return _of_kind(jobs, job_id, Job)


def _find_draft(jobs: JobStore, job_id: str) -> DraftJob:
    return _of_kind(jobs, job_id, DraftJob)


def _of_kind(jobs: JobStore, job_id: str, kind: type) -> Watched:
    """One store holds both sorts of job, so the id has to name the right one."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(
            404,
            "no such job. Jobs live in the memory of one process, so this means the server "
            "restarted, the job aged out, or the request reached a different worker than the "
            "one that started it -- run a single worker until jobs are persisted.",
        )
    if not isinstance(job, kind):
        raise HTTPException(404, "that job is not of this kind")
    return job


def _draft(job: DraftJob, open_session) -> None:
    """Bind one plan, on its own thread, appending bindings as they finish.

    The document is assembled once, at the end. A submission half-built is not a document anybody
    should be shown, and the page has the bindings as they land in any case.
    """
    try:
        model = build_structured(ScopeAssessment) if job.model_configured else None
        with open_session() as session:
            if not search.index_exists(session):
                search.build_index(session)
            for proposition in job.plan.propositions:
                job.add(proposition, authority_gate.bind_proposition(session, proposition, model))
            draft = assemble(job.plan, job.bindings)
            job.attacks = attack_draft(session, draft)
            job.document = to_markdown(draft, job.attacks)
        job.finish()
    except Exception as exc:  # noqa: BLE001 - the page needs to be told, whatever went wrong
        # The page is told, and so is the operator. The job id is the only identifier here: the plan
        # is the advocate's case and does not belong in a log.
        log.warning("draft job %s failed: %s", job.id, logs.reason(exc))
        job.finish(error=f"{type(exc).__name__}: {exc}")


def _preferred_citation(session, judgment_id: str) -> str | None:
    from orderorder.db.models import CitationAlias

    alias = session.scalars(
        select(CitationAlias)
        .where(CitationAlias.judgment_id == judgment_id)
        .order_by(CitationAlias.reporter, CitationAlias.citation_string)
    ).first()
    return alias.citation_string if alias else None


def _agent_status() -> dict:
    """Which model the agent would use, for the status line. Never raises.

    Reported on the health route because the agent silently falls back when AWS is not configured
    (see `orderorder.agent.model`), and a deployment that believes it is on Bedrock while every answer
    comes from a local 4B model looks exactly like one that is. The choice is made here rather than
    cached because a key can be set without restarting the process.
    """
    from orderorder.agent import choose_model

    try:
        return {"configured": True, **choose_model().as_json()}
    except Exception as exc:  # noqa: BLE001 - a status line must render whatever is wrong
        return {"configured": False, "reason": str(exc)}


def _run(job: Job, request: VerifyRequest, open_session) -> None:
    """Check one brief, on its own thread, appending verdicts as they finish."""
    try:
        use = job.model_configured
        with open_session() as session:
            verify_text(
                session,
                job.text,
                build_structured(ScopeAssessment) if use else None,
                voice_model=build_structured(VoiceAssessment) if use else None,
                weight_model=build_structured(WeightAssessment) if use else None,
                facts_model=(
                    build_structured(ApplicabilityAssessment) if use and request.facts.strip() else None
                ),
                matter_facts=request.facts,
                on_found=lambda count: setattr(job, "total", count),
                on_verdict=job.add,
            )
        job.finish()
    except Exception as exc:  # noqa: BLE001 - the page needs to be told, whatever went wrong
        log.warning("verification job %s failed: %s", job.id, logs.reason(exc))
        job.finish(error=f"{type(exc).__name__}: {exc}")


# The module-level application uvicorn imports by name, so `--reload` can re-import it. The token
# comes from the environment here because uvicorn constructs this one itself.
app = create_app(token=os.environ.get(TOKEN_ENV) or None)
