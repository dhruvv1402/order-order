# OrderOrder — deployment

| | |
|---|---|
| **Version** | 0.2 |
| **Date** | 9 September 2026 |
| **Companion documents** | [ARCHITECTURE.md](ARCHITECTURE.md) §10, §12, §13 · [TECH_STACK.md](TECH_STACK.md) · [ROADMAP.md](ROADMAP.md) §0 |

[ARCHITECTURE.md](ARCHITECTURE.md) §10 describes the topology this is heading for. This document is
narrower and more useful: what is actually built, what will stop you, and in what order to fix it.

**Deploying to AWS with the real corpus is §4**, which is §3's recipe with AWS nouns and the four
decisions that are specific to it: the region (ap-south-1, because that is where the open-data bucket
already is), EC2 rather than a container platform, one EBS volume rather than EFS, and where TLS
terminates.

---

## 1. What is ready and what is not

| | state |
|---|---|
| **Verification (Surface A)** | Deployable. Eight of twelve failure modes need no model at all, measured on held-out data, and a provider outage degrades to *not checked* rather than to a wrong answer. |
| **Search** | Deployable. Lexical, no model, 91% case@1 on a quoted line. Weak on paraphrases (33%) and that limit is documented rather than hidden. |
| **Drafting (Surface B)** | **Not for production.** The gate refuses to bind anything it has doubts about, which is a structural guarantee. Whether it *has* doubts about extent of support depends on the model, and a fast model gets that wrong. Ship it behind a flag, as an artefact to read, not a document to file. |

Deploy verification first. It is a product a lawyer can use this month, and the claim it makes is one
the numbers support.

---

## 2. The six things that will stop you

Two of them have since been dealt with and are kept here because the reasoning is still what you need
when you meet them: §2.5 is now wired rather than absent, and the operability half of §2.3 — a job
that hung with nothing said about it anywhere — is closed by a deadline and a log.

### 2.1 A model that answers — the only real blocker

Everything else on this list is an afternoon's work. This one is a decision.

| option | latency | verdict |
|---|---|---|
| Gemini free tier | 10 requests/minute | 18 minutes for a three-proposition draft. Unusable for serving. |
| `qwen3:4b`, local, CPU | **247 s per call**, measured | Proves the wiring. Not a service. |
| A routed gateway on free endpoints | 80% `429`, measured | Answers the health probe and then fails four calls in five. |
| A paid API tier | seconds | The only one that serves. |
| A retired model id | instant 404 | Worst of all: it does not read as a failure. See below. |

Three things to know before choosing.

The engine is **schema-bound everywhere**, so an endpoint that accepts requests but will not fill a
JSON schema is useless to it — `orderorder doctor --probe` makes one real call and tells you.

With a router in front, run `scripts/gateway-failure-rate.py --marker <file>` over a real workload: a
route can pass the probe and still be mostly rate-limited, which produces a report full of *not
assessed* that looks like data.

And **check that the model id still exists**, which sounds beneath mentioning and is the one that
actually cost this project a day. `gemini-2.5-flash` was the configured primary here and has been
retired: Google answers 404 for it on new keys and names `gemini-3.6-flash` as the replacement. It
failed in the worst possible way, and the reason is a direct consequence of the design being right.
Every model call in this engine degrades to *not assessed* rather than to a guess, so a **dead model
is indistinguishable from a corpus with nothing to say** — `evals/report-holdout-model.txt` scored
100% abstention and mode 4 at 0/14 and read like a result rather than like an outage. Nothing in the
verdicts was wrong; every one of them honestly said the model could not be reached. What was missing
was anywhere that said it *to an operator*, which is §2.3a. Probe before a run, probe again when a
report comes back emptier than the last one, and treat a model id as a thing that expires.

### 2.2 Exposure

`orderorder serve` binds `127.0.0.1` and asks for nothing, because on a single-user machine there is nothing to
protect. Bound anywhere else it **requires** `ORDERORDER_API_TOKEN` and refuses to start without one:

```bash
export ORDERORDER_API_TOKEN=$(openssl rand -hex 32)
orderorder serve --host 0.0.0.0
```

The compose `serve` profile declares the same variable with no default, so a stack brought up without
one fails while compose is still interpolating. `/api/health` is the one route that answers without a
token — a load balancer needs it and it gives nothing away.

Put a reverse proxy in front for TLS. The container publishes to `127.0.0.1:8000` deliberately, so
exposure stays an explicit decision.

### 2.3 Jobs live in one process

`web/jobs.py` is a dictionary. Verdicts stream to the page from the process that started the job, so
**run exactly one worker.** With two, requests land on whichever worker answers and job lookups 404 at
random. `orderorder serve` never passes `--workers`; the trap is running uvicorn directly.

A single process is fine for a pilot — a job is worthless once the tab closes, which is why it was
built this way. It is not fine for horizontal scaling, and the jobs table in ARCHITECTURE §4.13 is the
fix when that day comes.

**A job now has a deadline**, `MAX_JOB_SECONDS` in `web/jobs.py`, fifteen minutes. A Python thread
cannot be killed from outside, so the check happens between citations rather than during one: the
verdicts already paid for are kept and the next citation does not start. One wedged call is bounded
separately and at the right layer, by `REQUEST_TIMEOUT_SECONDS` in `engine/providers.py`. Without
both, a stalled provider left a job reading *in progress* for as long as the process lived, which a
reader cannot tell from slow. Eviction also prefers finished jobs now, so a long verification is not
dropped from under the person watching it because twenty newer jobs arrived.

### 2.3a It says something when it fails

There was no logging at all, which is a strange thing to find in something otherwise this careful,
and the reason it lasted is that every degraded check already reports itself *to the advocate*: the
verdict says the model could not be reached and grades the citation *not checked*. That is correct
and it is not operations. A provider that has started refusing every call produces a page full of
honest abstentions and, until now, no other trace.

So: one logger on stderr, where `docker logs` already looks. `ORDERORDER_LOG_LEVEL` sets the volume.
A failed model call is logged once, naming the whole fallback chain that failed and the reason; a
failed job is logged with its id. **Prompts are never logged** — a prompt carries the brief, an
uploaded brief is privileged, and the guarantee this deployment makes is that it is not stored, which
a log line would quietly undo. `tests/test_logs.py` is what holds that.

This is not an audit log. Nothing records who asked what, and §3a still says so.

### 2.3b The optional Chroma extra carries unpatched advisories

`uv sync --extra chroma` installs `chromadb` 1.5.9, which has **five open advisories and no fixed
version published**. It is an extra rather than a dependency precisely so the CI audit — which runs
against the locked production set — is not suppressed to accommodate it, and that is the right way
round. But the audit therefore does not see it, so nothing will warn an operator who enables it.

Nothing web-facing touches Chroma: `chroma-import` and `chroma-search` are CLI only, and the store is
a local directory under `ORDERORDER_DATA_DIR`, so the exposure is a local process rather than a
listening service. Enable it if something outside this process needs to query the vectors; do not
enable it on a box serving the API, and re-check for a fixed release before you do.

### 2.4 SQLite is one writer

The corpus is a 1.2 GB SQLite file. Reads are fine and concurrent; writes are not. Ingestion, the
full-text index rebuild and `orderorder embed` all write, so do not run them against a database that
is serving. `infra/docker-compose.yml` has Postgres with pgvector ready for when that matters —
`DATABASE_URL` switches it, and the corpus must be re-ingested rather than copied.

### 2.5 Migrations: wired, and there is one thing to get right on an existing database

`init_db()` calls `create_all()`, which creates missing tables and never alters an existing one, so
it cannot be how a deployed schema changes. Alembic now is: the environment lives in the package at
`src/orderorder/migrations`, so it ships in the wheel and is present in the container, and the
current schema is the baseline revision.

```bash
orderorder migrate            # empty database: build it
orderorder migrate --stamp    # a database that ALREADY holds the corpus: claim it, apply nothing
```

**Which of those to run is the whole decision.** A database ingested before migrations existed has
every table and no `alembic_version` row, so it looks to Alembic like a database nothing has been
applied to; running the baseline at it would try to create tables that are there and stop. `--stamp`
records that the baseline is already true, and the next real migration then applies cleanly. Run it
once, on the corpus you copied onto the box. `orderorder init-db` stamps what it creates, so a
database made from here on needs nothing.

The schema and the models are kept from drifting apart by the suite rather than by discipline:
`tests/test_migrations.py` asks Alembic the same question `--autogenerate` asks and fails if a model
has gained a column that no revision creates. Generate one with
`uv run alembic revision --autogenerate -m "what changed"`, from the repository root.

### 2.6 The resume list is held in memory, so a bigger corpus is a bigger box

Ingestion loads the judgments still needing text into a list before it fetches anything. A `Judgment`
ORM object measured **3.1 KB** on the Supreme Court corpus, so the list alone is:

| corpus | judgments | the resume list |
|---|---|---|
| Supreme Court 2013-2025 | 9,429 | 30 MB |
| **Supreme Court 1950-2025, what is held today** | **38,032** | **~118 MB** |
| Supreme Court 1950-2025 | ~50,000 | ~0.2 GB |
| One large High Court | ~1,000,000 | ~3 GB |
| All 25 High Courts | ~17,800,000 | ~56 GB |

Measured on 9 September 2026 against the live bucket: the 1950-2025 metadata holds **38,032**
judgments, not the ~50,000 the row above guesses, so its resume list is ~118 MB. Metadata for all
seventy-six years imports in about fifteen minutes. The text pass runs at about 65 judgments a
minute with the default eight workers, which puts a full-corpus text build near ten hours -- and
the run is resumable, so the estimate survives the machine it is measured on.

The text build has since been carried through: **38,005 of 38,032** judgments hold text
(707,647 paragraphs), and the 27 that do not are PDFs missing from the source itself, confirmed by
re-running `ingest bulk-text --retry` against them. Index, alias inference and the citator were
rebuilt over the full corpus afterwards.

Up to the whole Supreme Court this does not matter. Past it, ingest in batches: `--limit` now bounds
the *query* rather than slicing the list afterwards, so `--limit 20000` reads twenty thousand rows and
not the corpus, and the run is resumable, so repeating it walks the corpus a batch at a time. `--year`
does the same by year, and is the better handle when the source data arrives that way.

What is not yet fixed is that the list exists at all, and that `BulkResult` keeps one `Outcome` per
judgment for the closing report. Both are bounded by the batch rather than by the corpus once you use
`--limit`, which is why batching is the answer here rather than a rewrite. Streaming the resume list
by keyset and reporting incrementally is the change that removes the ceiling; it needs a decision
about ingestion order, which is currently newest-first and load-bearing for a demo.

The queue in front of the workers is already bounded and does not grow with the corpus: four
judgments per worker are submitted ahead, rather than one `Future` per judgment for the whole run.

**On disk**, measured on the 2013-2025 corpus — 9,429 judgments, 409,499 paragraphs, 1.22 GB. The
shares below are what generalise; the totals are not, since the corpus is now 707,647 paragraphs:

| | size | share | |
|---|---|---|---|
| `paragraph` | 493 MB | 40% | the text itself |
| `paragraph_fts_content` | 468 MB | 38% | **a second copy of the same text** |
| `paragraph_fts_data` | 144 MB | 12% | the inverted index, the part that does the work |
| indexes on `paragraph` | 89 MB | 7% | |
| everything else | 28 MB | 2% | judgments, aliases, opinions, citations |

That is **2.9 KB per paragraph**, of which 1.1 KB is the duplicate. The full-text index is a standalone
FTS5 table, so it keeps its own copy of every paragraph body. An external-content table
(`content='paragraph'`) would not, at the cost of a join to read a body back and of the `judgment_id`
column the queries currently take from the index for free. It is a real 38% of the database and it is
worth doing before the corpus is large, not after; it is not done here because it changes the shape of
the search query and every index built so far would need rebuilding.

Budget **~3 KB a paragraph** until then, and remember the corpus is one SQLite file: §2.4.

**None of this reaches the High Courts yet**, and that is a feature rather than a bigger disk.
`ingest/corpus.py` reads one bucket, `indian-supreme-court-judgments`. The High Court corpus is a
second public bucket in the same region whose keys carry a `court=` partition under the year that
nothing here composes, and whose years run back to 1950. Pointing `BUCKET` at it does not work. The
module docstring says what would have to change; PRD phase 3 is where it is planned.

---

## 3. A first deployment

```bash
# On the box, with the corpus already ingested onto a volume:
export ORDERORDER_API_TOKEN=$(openssl rand -hex 32)
export LLM_PRIMARY=... OPENAI_API_KEY=...          # or GOOGLE_API_KEY, GROQ_API_KEY

orderorder migrate --stamp                          # once, for a corpus built before migrations
docker compose -f infra/docker-compose.yml --profile serve up -d --build
curl -s localhost:8000/api/health                   # no token needed
curl -s -H "Authorization: Bearer $ORDERORDER_API_TOKEN" \
     "localhost:8000/api/search?q=a+misrepresentation+vitiates+consent"
```

**The image builds, and CI is where that is known.** It went unbuilt for as long as this document
existed, because the machine it was written on had no room for a Python base image. A runner has room,
so `.github/workflows/ci.yml` builds it on every push and then runs it: bound to `0.0.0.0` with no
token it must *fail to start*, with a token it must answer `/api/health`, `/api/search` must be 401
without one, and the security headers must be on the response. Those are the guarantees in
`web/auth.py` and `web/api.py` checked against the artefact that ships rather than against the source.

What has still never been done is running it against a real corpus volume. The container starts with
an empty database, which is enough to prove it boots and refuses correctly, and not enough to prove
ingestion works inside it.

**The corpus is not in the image.** It is 1.2 GB of state that outlives any version of this code, and
it mounts at `/data`. Build it on the box, or copy the SQLite file in, and check
`orderorder stats` before serving.

**Chown the volume, or nothing can write.** The image runs as uid **10001**, not root — a process
that cannot rewrite its own source is one exploit less. `/data` is created and owned by that user in
the image, so running with no volume works and a *named* volume inherits the right ownership. A
**bind** mount does not: the host directory's ownership is what the container sees, and a root-owned
one leaves the engine unable to write its own corpus. The failure arrives at the first request rather
than at boot, which makes it look like a bug in the engine.

```bash
sudo install -d -o 10001 -g 10001 /srv/orderorder-data     # before the first run
```

**Keys never enter a layer.** No `ARG`, no `COPY` of `.env`, and `.dockerignore` excludes it — a
secret baked into an image is published to whoever can pull it, and `docker history` shows it even
after a later layer deletes the file.

---

## 3a. The security posture, item by item

Against the standard web-application checklist. The honest half of this table is the **not
applicable** column: a control that does not apply is not a control that has been passed, and writing
"n/a" without a reason is how a gap gets filed as a pass.

The shape of the thing decides most of it. There are no accounts, no passwords, no sessions and no
cookies; there is one bearer token that the operator generates, and one process that reads a corpus
of public judgments. What is genuinely sensitive is the brief an advocate uploads, and the answer for
that is that it is never stored.

| | control | where |
|---|---|---|
| 1 | **Keys are never in the image or the repo.** No `ARG`, no `COPY` of `.env`, `.dockerignore` excludes it; they arrive as environment at run time. `docker history` would show a key baked into a layer even after a later layer deleted it. | `Dockerfile`, `.dockerignore` |
| 2 | **History is scanned for secrets**, whole history rather than the diff — a key committed and later deleted is still published. | `.github/workflows/ci.yml` |
| 6 | **Authorisation is server-side and central**, in one middleware ahead of every route rather than per-route. A per-route check is a check somebody forgets on the route added next week, and the route added next week is the upload endpoint. | `web/api.py`, `web/auth.py` |
| 6 | **The binding decides.** Loopback asks for nothing; bound anywhere else a token is *required* and the server refuses to start without one, rather than coming up open with a warning nobody rereads. | `web/auth.py` |
| 11 | **Failed authentication is rate-limited hard** (5/min), separately from ordinary traffic. | `web/limits.py` |
| 12 | **Ordinary traffic is rate-limited too**, and starting work harder than reading a result — the resource being protected is one worker, not a bill. `X-Forwarded-For` is deliberately not trusted. | `web/limits.py` |
| 13 | **Every query is parameterised.** SQLAlchemy throughout; the two places that use raw SQL bind their parameters and interpolate only a module constant for the table name. | `engine/search.py` |
| 14 | **Input is validated and bounded** — Pydantic models on every body, a character cap on briefs and plans, and a declared-length cap refused before the body is parsed. | `web/api.py` |
| 15 | **User content is escaped at the last step before the DOM**, covering `& < > " '`. | `static/app.js` |
| 16 | **Uploads are bounded and named.** The read stops at 25 MB rather than checking the size after spending the memory, and only a brief's formats are accepted. | `web/api.py` |
| 17 | **Responses carry what the page needs and no internals.** A 401 does not say which half was wrong; a failed parse returns the exception's *type*, never its message or a path. | `web/api.py` |
| 18 | **Security headers on every response, refusals included** — CSP with no `unsafe-inline`, nosniff, DENY, no-referrer, and the cross-origin isolation pair. | `web/api.py` |
| 20 | **Dependencies are audited** on every push and weekly, against the locked production set. | `.github/workflows/ci.yml` |

| | not applicable, and why |
|---|---|
| 3 | **No public database key.** Nothing in the browser talks to a database. The page calls this API and this API alone; the corpus is a file the server opens. |
| 4 | **No row-level security**, because there are no rows belonging to anyone. The corpus is public case law, identical for every reader, and read-only over HTTP. Uploaded briefs never reach a table — they are parsed in memory and handed straight back. The day matters become multi-tenant this becomes the first item on the list, and ARCHITECTURE §4.13 is where it starts. |
| 5 | **No encryption at rest of "sensitive data"**, because the data at rest is published judgments. The genuinely privileged thing is the brief, and it is not stored at all, which is a stronger guarantee than encrypting it would be. Disk encryption on the host is the operator's call and the right layer for it. |
| 7, 8 | **No per-record access control or field-tampering surface.** There is no record a user owns and no field a user can write. Every route is read-only against the knowledge base. |
| 9, 10 | **No cookies, sessions or passwords.** Authentication is one bearer token the operator generates and compares with `hmac.compare_digest`. Nothing to hash, nothing to set `Secure` and `SameSite` on. |
| 19 | **HTTPS is not forced here, deliberately.** This binds to loopback and expects a reverse proxy to terminate TLS (§2.2). A service that cannot see whether it is on HTTPS should not be the one sending HSTS — that header belongs in the proxy, and asserting it from behind one that is misconfigured locks users out of a host that never served TLS. |

What is **not** covered and should be said plainly: there is no audit log of who asked what, no CSRF
token (the API is token-authenticated and takes no cookies, so a browser cannot be made to speak for a
user it is not already carrying a token for), and the rate limiter counts in one process, so it is
exactly as durable as the process — a restart forgives everyone. All three are consequences of the
single-process design in §2.3 and change together with it.

---

## 4. On AWS, with the real corpus

One fact decides most of this section. The corpus comes from `indian-supreme-court-judgments`, an
AWS Open Data bucket in **ap-south-1**, read over anonymous HTTPS with no credentials and no boto3
(`ingest/corpus.py`). So run in **ap-south-1**: the ten-hour text build then reads from a bucket in
its own region, which is both fast and free of transfer charge, and Mumbai is where an Indian legal
corpus and anything an advocate uploads should be sitting anyway.

### 4.1 EC2, not ECS or Fargate

This wants to be a container platform and it is not one, for reasons already in this document rather
than any AWS-specific ones:

- The corpus is **one SQLite file** and SQLite is one writer (§2.4).
- **Jobs live in one process** (§2.3), so exactly one worker — which means no horizontal scaling, no
  target group with two healthy targets, and no autoscaling group above one. Sticky sessions do not
  rescue this: a job id is held in a dictionary in one process, and a request that lands anywhere else
  gets a 404.
- The corpus is **ten hours of work**, so it must outlive a task restart. Fargate's ephemeral storage
  does not.

**Do not put the corpus on EFS.** SQLite's locking over NFS is the classic way to corrupt a database
that was working fine, and the read pattern here — an FTS5 index and a memory-mapped matrix — is the
worst possible fit for a network filesystem. The corpus wants a block device: **one EBS gp3 volume**.

So the shape is one EC2 instance running the compose `serve` profile, with EBS at `/data`. That is
§3's recipe with AWS nouns.

### 4.2 Two instance sizes, because building and serving are different jobs

The text build runs at about **65 judgments a minute on eight workers**, which is near ten hours for
38,032, and it is compute-bound — PDF extraction and cleaning, eight worker processes each holding a
judgment. Serving is one mostly-reading process. Sizing one instance for both means paying for the
build's cores for the lifetime of the deployment.

So: build on a compute instance, **snapshot the EBS volume**, then serve from a small one and attach
the snapshot. The snapshot is the backup as well — ten hours is worth not repeating, and it turns a
rebuild into minutes.

**Volume size.** Measured at ~2.9 KB a paragraph (§2.6), so 707,647 paragraphs is about **2 GB**, plus
340 MB if you run `orderorder embed`, plus headroom for an index rebuild, which wants room for both
copies at once. **20 GB gp3** costs little enough to stop thinking about and is large enough that you
never have to.

**Memory** is the build's constraint rather than the serve's: eight workers, plus a resume list that is
~118 MB at this corpus size (§2.6). If the instance is tight, `--limit` bounds the query and the run is
resumable, so batching down is always available.

### 4.3 Ingest on the instance, not on a laptop

Both halves of this matter. The bucket is in-region, so the download is fast and free. And a 2 GB
SQLite file pushed up from a laptop over a domestic connection is slower than rebuilding it in-region
from scratch.

Run it under `tmux` or `screen`, or as a systemd unit. Ten hours outlives an SSH session, and although
the run is resumable, discovering that at hour nine is a bad way to learn it.

```bash
# On the build instance, with the volume mounted at /srv/orderorder-data and owned by uid 10001.
export ORDERORDER_DATA_DIR=/srv/orderorder-data

orderorder init-db
orderorder ingest metadata $(seq 1950 2025)   # about fifteen minutes for all seventy-six years
orderorder ingest bulk-text                   # about ten hours; resumable, so re-run it after a drop
orderorder ingest bulk-text --retry           # confirm the stragglers are missing at source
orderorder ingest aliases
orderorder ingest repair-trailers
orderorder index
orderorder citator
orderorder stats                              # 38,032 judgments before you believe any of it
```

`orderorder embed` is optional and dense retrieval is off by default (§11.4 of ARCHITECTURE); skip it
unless you intend to run with `--dense`. If you do want it, use `orderorder embed --api` against a
hosted `/v1/embeddings` endpoint rather than the local encoder — it is about $3 for the corpus, it is
the stronger model, and it is not what you want the build instance's cores doing for an extra hour.

Rhetorical roles are worth running here even though they cost minutes: `orderorder ingest mark-roles`
after the text build. Retrieval reads them as a filter, and a corpus without them simply has that
filter return nothing.

**Ingestion inside the container is untested** (§3). `docker compose run --rm api ingest ...` should
work — the entrypoint is `orderorder` and only the command is `serve` — but nobody has done it against
a real corpus. Running the CLI directly on the build instance avoids finding out the hard way, and is
what the commands above assume.

### 4.4 Exposure, and where TLS terminates

The container publishes to `127.0.0.1:8000` deliberately (§2.2), and the binding rule means a token is
required the moment it is anything else. Two shapes work:

| | how | when |
|---|---|---|
| **Reverse proxy on the instance** | Caddy or nginx terminating TLS, proxying to `127.0.0.1:8000` | The default. Nothing in the compose file changes, and the published port stays on loopback |
| **ALB in front** | Publish to the instance's private address instead of loopback, and lock the security group so **only the ALB's security group** can reach the port | When you want ACM certificates, WAF, or access logs |

The ALB route trades the loopback guarantee for AWS-managed TLS, so the security group becomes the
thing standing between the corpus and the internet. **Never `0.0.0.0/0` on port 8000.** The token is
still required in both shapes — it is not an alternative to the network control, and neither is an
alternative to the other.

### 4.5 Secrets

`ORDERORDER_API_TOKEN` and every model key belong in **SSM Parameter Store (SecureString)** or Secrets
Manager, fetched at container start by an instance role. Not in the AMI, not in the compose file, not
in an image layer — `docker history` shows a key baked into a layer even after a later layer deletes
it (§3a, item 1).

With the multi-account fallback chain, each account's key is its own parameter:
`/orderorder/GOOGLE_API_KEY`, `/orderorder/GOOGLE_API_KEY_2`, and so on, matching the `#VARIABLE`
names in `LLM_FALLBACKS`.

Give the instance role the minimum: read those parameters, and write its own CloudWatch log group.
Nothing here needs S3 credentials — the open-data bucket is anonymous.

### 4.6 What the region does and does not buy

ap-south-1 keeps the corpus, the database and anything uploaded inside India, which is what the DPDP
position in [PRD.md](PRD.md) §15 wants. Uploaded briefs are never stored at all (§3a, item 5), which is
stronger still.

**It does not decide where the model runs.** A brief's text leaves the region the moment it is sent to
a hosted provider, and no AWS setting changes that — `LLM_SENSITIVE` is the control, naming the one
provider allowed to see text that is not demo data. If privilege is the concern, the answer is the
self-hosted production profile in [ARCHITECTURE.md](ARCHITECTURE.md) §10, not a region.

### 4.7 The order to do it in

1. Launch a compute instance in **ap-south-1**; attach a 20 GB gp3 volume.
2. `sudo install -d -o 10001 -g 10001 /srv/orderorder-data` — before anything writes there. A
   root-owned bind mount fails at the *first request* rather than at boot, which reads as an engine bug
   (§3).
3. Ingest, per §4.3. Check `orderorder stats`.
4. **Snapshot the volume.** This is the artefact worth protecting.
5. `orderorder migrate --stamp` — once, because the corpus was built before migrations were applied to
   it (§2.5).
6. Downsize: launch the serving instance, attach a volume from the snapshot.
7. Put the token and keys in Parameter Store; `docker compose --profile serve up -d`.
8. `orderorder doctor --probe` — one real call, confirming a filled schema comes back. A route that
   answers but will not fill a schema degrades every model-dependent check to *not assessed* (§2.1).
9. `curl -s localhost:8000/api/health`, then a token-bearing `/api/search` against the real corpus.
10. Reverse proxy and TLS. Only then a DNS record.

### 4.8 Costs, honestly

The shape is: a compute instance for about a day, then a small always-on instance, 20 GB of gp3, one
snapshot, and whatever the model tier costs — which will dominate all of it. Model spend is the real
line item and §2.1 is the argument for a paid tier.

The pricing pass in [TECH_STACK.md](TECH_STACK.md) §12.2 is from September 2026 and was aimed at GPU
boxes for the self-hosted profile. **It does not cover this deployment and should not be read as
though it does**; verification needs no GPU at all. Price the two instance sizes against current
ap-south-1 rates before committing to them rather than trusting a figure written here.

---

## 5. Before anyone else uses it

- **Attribution.** Judgment data is CC-BY-4.0 from AWS Open Data. The attribution is in the README and
  belongs anywhere the corpus is served.
- **The disclaimer is not decoration.** Every export carries it and every claim about a citation is
  three-state: supported, checked-and-not-supported, or *not checked*. A deployment that flattens that
  third state into either of the others is the failure this whole engine exists to prevent.
- **Uploaded briefs are privileged.** Nothing is sent anywhere except the model calls, and
  `LLM_SENSITIVE` names the provider allowed to see text that is not demo data. Honour it.
- **Rate limits are a correctness problem, not just a speed one.** When a provider fails, the engine
  records *not assessed* — correct, and it means a throttled deployment quietly checks less than it
  appears to. Watch the abstention rate.
