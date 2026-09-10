# OrderOrder

*A self-hosted citation-integrity engine for Indian case law.*

OrderOrder reads a brief, moot-court memorial or written submission, finds every case-law citation, and answers three questions separately for each one: does the case exist, which paragraph is being relied on, and does that paragraph support the proposition **to the extent claimed**. It then writes what opposing counsel would say. The same engine gates a drafting assistant so that nothing enters a written submission that the verifier could not confirm.

It is built on open and official data (AWS Open Data judgments under CC-BY-4.0, the Supreme Court's SCR portal, the Indian Kanoon API with attribution). The hackathon build costs nothing and runs on free tiers with demo data only; in production everything runs on hardware the team controls.

## Documents

| Document | Read it for |
|---|---|
| [docs/PRD.md](docs/PRD.md) | The problem, the evidence that it is urgent, users, the twelve ways a citation lies, requirements for both surfaces, metrics, competition, risks, compliance |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | What is ready to deploy, the six things that will stop you (two now closed), a first deployment, the security posture item by item, and the AWS runbook for the real corpus |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design with nine diagrams: ingestion, the verification engine, the drafting engine, retrieval hierarchy, data model, verdict schema, deployment, evaluation |
| [docs/TECH_STACK.md](docs/TECH_STACK.md) | The zero-cost hackathon stack and the self-hosted production stack, how LangChain and LangGraph are used, free-tier limits and data terms, licence audit, dev-machine setup, bill of materials |
| [docs/ROADMAP.md](docs/ROADMAP.md) | The 10-day hackathon sprint with demo script and cut list, then the startup phases, team split, decision log |

Suggested reading order: PRD §1-6, then ARCHITECTURE §1-4, then TECH_STACK §4, §5 and §12, then ROADMAP §1.

## Running it

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. The environment script points every
cache, the interpreter and the virtualenv at one data directory, so that the corpus, the model
caches and the build caches — several gigabytes between them — stay off the system drive. Set
`ORDERORDER_DATA_DIR` to choose where; it defaults to `./data`.

```bash
source scripts/dev-env.sh        # PowerShell: . .\scripts\dev-env.ps1
uv sync                          # installs into $ORDERORDER_DATA_DIR/venv
cp .env.example .env             # optional; defaults work for everything below

uv run orderorder doctor                          # check the environment
uv run orderorder init-db                         # create tables (SQLite by default)
uv run orderorder migrate                         # or bring an existing database up to the schema
uv run orderorder ingest metadata 2019 2020 2021  # download and import a few years
uv run orderorder stats

uv run orderorder cite parse "Kesavananda Bharati v. State of Kerala, (1973) 4 SCC 225, para 316"
uv run orderorder resolve "[2019] 9 S.C.R. 593"

uv run orderorder ingest text INSC:2019:770          # fetch the official PDF, clean and segment it
uv run orderorder locate INSC:2019:770 "the plaintiff is the dominus litis" --pinpoint 73

uv run orderorder verify --file brief.txt            # every citation in a brief
uv run orderorder verify --file brief.txt --facts matter.txt   # ... and whether each one applies
uv run orderorder verify --file brief.txt --memo     # ... and what the other side will say
uv run orderorder verify --file brief.txt --annotate flagged.txt --report report.md

uv run orderorder ingest bulk-text                   # text for the whole corpus; resumable
uv run orderorder ingest aliases                     # learn the SCC citations the open data omits
uv run orderorder ingest repair-trailers             # cut the reporter's own words out of stored text
uv run orderorder ingest mark-roles                  # label every paragraph's rhetorical role, by cue
uv run orderorder index                              # full-text index over every paragraph
uv run orderorder citator                            # who cited whom, and what they did with it
uv run orderorder embed                              # vectors for every paragraph (see the numbers first)
uv run orderorder embed --api                        # ... or through a hosted endpoint: BGE-M3, ~$3
uv run orderorder chroma-import                      # optional: the same vectors into a local Chroma db
uv run orderorder chroma-search "notice under Section 106"   # ... and query it directly
uv run orderorder find "a misrepresentation vitiates consent only where it induced the contract"
uv run orderorder contrary "a notice under Section 106 is mandatory before a suit for eviction"
uv run orderorder argue propositions.txt             # bind each proposition to an authority, or refuse
uv run orderorder draft demo/plan.txt --docx out.docx  # assemble a written submission from a case plan
uv run orderorder treatment INSC:2019:770            # is this authority still good law?

uv run orderorder eval generate --seeds 40 --rng-seed 1729   # plant known failures in real judgments
uv run orderorder eval run --no-model --detail               # score every check that needs no model
uv run orderorder eval search --judgments 40                 # score the other direction
uv run orderorder eval gate --judgments 40 --no-model        # what the drafting gate is filtering
uv run orderorder eval contrary --judgments 40               # one holding put both ways

uv run orderorder serve                              # all of it in a browser, on localhost
```

The engine runs in two directions, and then reads the second one for the opposite sign.

`verify` starts from a citation the brief already gives. It resolves it, loads the judgment, ranks the
paragraphs, asks how far they support the claim, works out whose words the relied-on paragraph carries
and whether they decided anything, and grades the result. Extent of support and ratio-versus-obiter
need a language model; the other checks do not, and the command says so rather than staying silent.

`find` starts from a proposition and no citation, which is where a lawyer preparing arguments actually
starts. It searches every paragraph of every judgment held, drops anything that is not the court
speaking, weighs bench strength and recency beside relevance, and names the sentence to read. With a
model configured it then runs each authority back through `verify`, because retrieval proposes and
only the verifier confirms.

`contrary` asks the question the other two cannot: not which judgment supports you, but which one says
the other thing. The instinct is that a contrary holding is far away in the retrieval field and needs
some other kind of search to reach, and it is the opposite — a paragraph that contradicts a
proposition is the *nearest* text in the corpus to it, because it is about the same subject in almost
the same words. "A notice under Section 106 is mandatory" and "the requirement of notice under Section
106 is directory" share every distinctive term. BM25 cannot separate them and neither can a dense
encoder trained to put sentences about one subject in one place. What separates them is polarity,
which is grammar rather than ranking, so the retrieval is the ordinary one and all the work is in what
happens to the field afterwards: whether the negation in a court's sentence governs the clause the
proposition is about, or the condition attached to it, or something else in the same breath. What
comes back is a lead — a court, in its own voice, wrote a sentence on this subject with the opposite
sign — and the fix it carries is "read it", never "drop the point".

`argue` is the two of them composed, and is the drafting surface's gate. For each proposition a
lawyer intends to advance it searches, verifies, and then refuses everything that is not the court's
own words, from the majority, still good law, and quoted verbatim. Where the court put the point more
narrowly than the advocate did, it returns the authority *and the proposition to argue instead*.

`draft` takes a case plan — the court, the parties, the issues, the propositions an advocate intends
to argue, the prayer — and assembles a written submission in the Indian filing format, with `argue`'s
gate deciding what may go behind each sentence. A proposition that found no authority is not dropped:
it stays where it was written, marked, because a draft that quietly loses its unsupported sentences
reads as though everything in it is supported. The list of authorities is built from what was verified
and can contain nothing else, and an appendix gives the paragraph and the verified words behind every
citation in the document, so a supervisor can check the whole thing against the reports without
running any of this. It writes Markdown and a .docx that opens in Word.

It then attacks its own draft. Everything in the document has already passed the gate, so the useful
attacks are the ones the gate cannot make: a case a later court held inapplicable on its facts is
still good law and still the argument you will meet; a larger or later bench sitting on the same words
that the draft does not cite is the first thing the other side's junior will find; an authority no
later judgment has cited at all is something they will say out loud. All of it comes from the citation
graph and the bench strengths, so no model is called and nothing in the section can be an invention.
It also says what it did **not** look at, because a self-attack that stops at what it found reads as
an assurance.

`serve` puts the three outputs of a stress-test on one page, which is the point of having a page.
A brief can be pasted or opened from a PDF or DOCX; what is read is shown before it is checked, so a
mangled extraction is obvious in a second rather than arriving as nine inexplicable phantoms. Then:
click a flag, see the paragraph the engine read, see the verified sentence highlighted inside it.
The board fills as each citation is decided, so a phantom is on screen while the ones that need a
model are still running. The third tab is the drafting workspace: paste a case plan, watch each
proposition bind or refuse as it is decided, read the assembled submission and what the other side
will say about it, and download the .docx. It binds to localhost; the corpus and anything pasted into
it stay here.

`uv run pytest` runs the suite; it uses an in-memory database and never touches the network.

### On a box that is not a development machine

```bash
docker build -t orderorder .
ORDERORDER_API_TOKEN=$(openssl rand -hex 32) \
  ORDERORDER_DATA_DIR=/srv/orderorder-data \
  docker compose -f infra/docker-compose.yml --profile serve up -d
```

Two things stay outside the image, and both are deliberate. **The corpus**, because 1.2 GB of SQLite
is state that outlives any version of this code and baking it in would mean rebuilding the image to
ingest one judgment; it arrives as a volume at `/data`. **Keys**, because a key in a layer is a key
published to everyone who can pull the image, and `docker history` shows it even after a later layer
deletes the file; they arrive as environment at run time.

The token is required and has no default. A container publishes a port by definition, so the
in-process rule — bound to anything but loopback, a bearer token or the server does not start — is
what stands between a `docker compose up` and nine thousand judgments, an upload endpoint and a
billable model key answering to whatever can route to the host. Compose substitutes an empty string
for a variable nobody set, an empty token is no token, and `serve` refuses to boot. The stack fails
at start rather than coming up open, which is the whole intent: an open server looks exactly like a
closed one until somebody finds it.

The published port is `127.0.0.1:8000`. Anything beyond that host wants a reverse proxy in front,
so that TLS is somebody's decision rather than a port that happened to be open.

### What it scores

The engine is measured in all three directions, against ground truth the corpus supplies rather than
labels anyone wrote, on **forty judgments the detectors were not developed against**. The set they
were fixed on cannot measure them, so every number here is from a held-out draw.

Verification, 270 planted items, **no model configured**:

| mode | | recall |
|---|---|---|
| 1 | phantom | 40/40 |
| 2 | mis-cite | 40/40 |
| 3 | wrong court or bench | 39/39 |
| 5 | wrong voice | 34/34 |
| 9 | selective quotation | 20/20 |
| 10 | dead or wounded law | 14/14 |
| 12 | wrong pinpoint | 40/40 |
| | **false positives on clean citations** | **0/40** |

With a model, over the modes that need one and 25 clean citations: obiter as ratio 3/3, false
positives 0/25, **quote grounding 100%**, abstention 68%, six seconds a citation.

Six seconds is a hosted model. The same checks against `qwen3:4b` run locally on four CPU cores with
no usable GPU take **247 seconds a call**, measured. The model is not the problem: every answer came
back as a filled schema and every quote verified. The hardware is. Prompt processing ran at 8-9
tokens a second cold against 5 for generation, which is backwards, and the cause is memory: once the
weights are loaded there is not enough left to hold the prompt, so the OS pages it and two thirds of
each call is the model reading its own input back off disk. A local model on a machine like that is
therefore good for `doctor --probe` and a handful of propositions, and not for a brief.

Search, 148 queries, measured on the 2013-2025 corpus (409,499 paragraphs):

| query | case@1 | case@5 | para@5 | line |
|---|---|---|---|---|
| the line, verbatim | 91% | 99% | 99% | 100% |
| a remembered fragment | 81% | 91% | 85% | 100% |
| a paraphrase | 33% | 44% | 36% | 94% |

Once the right paragraph is found, the sentence named as the line is the one the proposition came from
almost every time. Getting to the right paragraph is another matter, and the third row says why:
retrieval is lexical, so it finds the judgment's own words and not an idea restated in someone else's.

**Re-measured on the full corpus** (668,272 paragraphs with vectors, `potion-base-8M`), which is also
where the dense half was finally settled. Only case@1 and the paraphrase depths were re-run:

| query | lexical only | + dense, 4 votes | + dense, 1 vote |
|---|---|---|---|
| verbatim @1 | **98%** | 86% | 96% |
| fragment @1 | **85%** | 43% | 76% |
| paraphrase @1 | 25% | 33% | **30%** |
| paraphrase @5 | 42% | 40% | **47%** |
| paraphrase @10 | 48% | 40% | **56%** |

That table is a better answer than the one it replaces, and it corrects it. The earlier reading was
that dense retrieval simply made things worse; on the full corpus at **one** vote rather than four it
plainly helps the row it was bought for — paraphrase recall goes 42% to 47% at five and 48% to 56% at
ten. What it charges for that is nine points off the fragment row, and a lawyer typing a half-remembered
line is not a rare user. So `dense` still defaults to off, but for a reason that is now a trade rather
than a verdict: **a static 8M model buys five points on paraphrases and charges nine on fragments.**
`--dense` turns it on. A stronger encoder on a GPU is the experiment that can change the trade; a
bigger CPU model is not, per the discrimination test in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §11.4.

Note also what the bigger corpus did on its own: verbatim recall went **up** (91% to 98%) and
paraphrase recall went **down** (33% to 25%). More candidates make an exact string easier to pin and
an idea harder, which is the same fact from both ends.

Drafting, 74 propositions lifted from the same forty judgments — half of them from paragraphs
reciting counsel's argument rather than the court's holding:

| the proposition was lifted from | paragraphs a word search returns | of those, not the court speaking |
|---|---|---|
| the court's own words | 4.0 | 0.5 |
| counsel's submission | 4.0 | **2.2** |

That second row is why the drafting surface has a filter and a gate at all. Ask the corpus for a
proposition phrased the way an advocate phrases one — which is to say baldly, without the
qualifications a court attaches — and **more than half of what a word search hands back is somebody's
argument rather than anybody's holding.** It reads like better authority than the holding does,
because that is what an argument is for. The source paragraph itself was within reach in **34 of 34**
cases, and none of the 34 survived to reach the gate.

What that measurement cannot do is tell you the voice detector is right: the same detector labels the
item and drops the paragraph. What it does tell you is the size of the field it is working over, which
is a fact about the corpus rather than about the detector. `evals/report-gate.txt` has the run.

The contrary search, 40 holdings from the same forty judgments, each put to the engine twice — once
negated, which is what the other side argues, and once as the court wrote it, which is what the side
relying on it argues. Same paragraph, same retrieval; the only difference is the polarity of the
sentence put to it:

| the proposition put | the source paragraph was retrieved | it was called contrary | leads returned |
|---|---|---|---|
| the holding, negated | 100% | 55% | 1.4 |
| the holding, as written | 100% | **5%** | 0.7 |

The gap between those two middle numbers is the measurement, and it is the only one here that is not
circular: the negation that builds the query and the negation the detector reads are the same idea, so
the 55% is a property of the construction and is printed because a low one would mean something broke.
Twenty of the forty were called contrary for the negated proposition and not for the court's own
words; **none** went the other way. The 0.7 is the floor — passages offered against a proposition that
was never in doubt, an upper bound rather than a count, since courts do disagree and this corpus holds
nine thousand of them. `evals/report-contrary.txt` has the run and the leads themselves, which is the
part worth reading: what is left is a lead to check, and the report says so rather than calling it a
contradiction.

That floor is half what it was, because shared terms are now weighed by how rare they are in the
corpus rather than counted. A proposition and a passage whose whole overlap is a stock phrase — "suit
for specific performance", which is in thousands of paragraphs — are no longer treated as being about
the same subject. Counting instead of weighing returns a spurious lead for 24 of the 40 propositions
against 12, and 1.5 leads each against 0.7. The price is one true detection in twenty-one, and no time
worth reporting: the weights are one index count per term, and that cost was smaller than the
variation between two runs of the same configuration.
[`evals/report-contrary-unweighted.txt`](evals/report-contrary-unweighted.txt) is that run, so the
trade is checkable rather than asserted; `orderorder eval contrary --no-weighted` reproduces it.

What these numbers do not say — and the limits matter more than the figures — is set out
in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §11.4.

### What works today

All three questions the product asks about a citation: **does the case exist**, **which paragraph is
being relied on**, and **does that paragraph support the claim to the extent claimed** — and, once a
paragraph is fixed, **whose words they are** and **whether they carried the decision**. Between them
these detect **all twelve** failure modes in the taxonomy in [docs/PRD.md](docs/PRD.md). Eight of the
twelve are decided without a language model at all, which is why the whole of the demo above runs on
an ordinary CPU with no key configured.

And the same machinery run backwards: **given a proposition and no citation, which judgment backs it,
and which line**. Search drops any passage that is not the court speaking before it ever reaches the
lawyer, so the tool cannot suggest as authority the kind of passage the verifier exists to catch.

And the same retrieval read for the opposite sign: **which judgment says the other thing**. No model
is involved and none is needed to find them, because what distinguishes a contradiction from a
restatement is the polarity of a clause and not the ranking of a paragraph.

**The model is never believed, only checked.** It is asked to name a paragraph and copy a sentence
from it. That sentence is then string-matched against the stored judgment. A quote that does not match
downgrades the claim to unsupported and flags it, however confident the model was. Born-digital text
never fuzzy-matches, so a near-miss paraphrase fails too. This is ordinary Python and it runs whatever
model produced the answer, which is why the tests can exercise it with a stub and no API key.

**Two ways a check can be confidently wrong, both now closed.** A brief that recounts what happened
below — "Rejecting the plea, the High Court opined that ..." — is not claiming the High Court is the
authority it relies on; it is narrating the history of the case it cites. Read naively, every such
clean sentence became a wrong-court finding. A fronted procedural participle before the court phrase
now marks it as a recital, and a genuine attribution ("the High Court has held that X") carries no
such participle. And a cause title that names nobody — "State of U.P. v. Anr." — token-matches
hundreds of judgments across seventy-five years at a *passing* score, so the citation alone picked the
case and the name check silently agreed. That is now a **request for review** rather than a pass: not
a finding, because nothing is wrong yet, but a question the strings cannot answer and so must not
appear to have answered. Both were caught by the held-out set, which is what a held-out set is for.

Three states are kept apart, because collapsing them is how tools overclaim: **supported**,
**checked and not supported**, and **not checked**. A missing API key, a provider outage or a
retrieval miss produces the third, never the second.

**A passage inside a judgment is not automatically the court's holding.** It may be counsel's
submission recited by the bench, the court below being quoted, an earlier judgment quoted, the
publisher's headnote, or the dissent. Whose words they are is decided from the judgment's own
structure and from attributing cues in the text — never from a model — and the cue that governs is the
last one before the sentence actually relied on, because a paragraph routinely sets out an argument
and then rejects it. The finding names the cue, so a reader can check it against the judgment. This
check needs no API key: a brief pinpointing a dissent is caught with nothing configured at all.

Ratio and obiter are told apart the same way where the court says so in terms ("it is not necessary
for us to decide"), and otherwise by a model whose answer must quote the sentence that shows it. An
answer that cannot be grounded becomes `unclear`, which costs a citation nothing. Calling a holding a
passing remark is as damaging as the reverse, so the classifier abstains rather than guesses.

**And a citation can be sound in every one of those respects and still be dead law.** The citator
reads every judgment's citations of every other, and what the citing court did with each: followed,
distinguished, doubted, overruled. One rule there is arithmetic rather than language — a bench cannot
overrule one at least as large as itself, so two judges saying a three-judge decision "does not lay
down the correct law" are recorded as having doubted it, with the claim attached. Reporting an
overruling that did not happen would have an advocate drop a binding authority.

The treatment report always states how many judgments it searched, because "no negative treatment
found" over nine thousand judgments means something different from the same words over the full
seventy-five years. On the 2013-2025 corpus, **56% of the citations these judgments make were to cases
decided before 2013** — the single largest hole in what the citator could see. Ingesting 1950-2025
closes most of it by construction. How much is a number nobody has taken yet: the citator was rebuilt
over the full corpus and its edge count has not been reported, so the honest statement is that the
gap is now small rather than that it is a particular size.

| Piece | Module |
|---|---|
| Citation grammar for SCC, SCC OnLine, AIR, SCR, SCALE, JT, INSC, High Court neutral citations and Indian Kanoon IDs, with pinpoints and party names | `citations/grammar.py` |
| Corpus client for the AWS Open Data bucket (anonymous HTTP, no AWS account) | `ingest/corpus.py` |
| Metadata import into `judgment` and `citation_alias` | `ingest/metadata.py` |
| Resolver: exact alias match, then fuzzy party names | `resolver.py` |
| Reports PDF cleaning: margin letters, running headers, the editorial headnote, the coram and the authoring judge | `ingest/pdf.py` |
| Paragraph segmentation with printed labels, sub-labels and offsets | `ingest/segment.py` |
| Persistence of text versions, opinions and paragraphs | `ingest/store.py` |
| BM25 ranking inside a judgment, and the fusion seam for embeddings | `engine/lexical.py` |
| Locator and pinpoint checking | `engine/locator.py` |
| Quote verifier, the quote-or-nothing rule | `engine/quotes.py` |
| Scope comparator: extent of support, dropped conditions, a narrowed proposition | `engine/scope.py` |
| Voice and opinion: counsel's argument, the court below, a quoted precedent, the headnote, the dissent | `engine/voice.py` |
| Weight: ratio versus obiter, with abstention as the default | `engine/weight.py` |
| Sentence boundaries that survive "Kasturi v. Iyyamperumal" and "[2019] 9 S.C.R. 593" | `engine/sentences.py` |
| Corpus-wide authority search: which judgment backs a proposition, and which line | `engine/search.py` |
| The opposite direction: which judgment says the other thing, by clause polarity rather than by ranking | `engine/contrary.py` |
| The citator: treatment of one judgment by later ones, and the bench-strength rule | `engine/citator.py` |
| Court and bench attribution: what the brief claims against what the record says | `engine/hierarchy.py` |
| Applicability: whether the cited case governs the facts of this matter | `engine/facts.py` |
| Resumable bulk text ingestion for the whole corpus | `ingest/bulk.py` |
| Learning the reporter citations the open data omits, from how judgments cite each other | `ingest/aliases.py` |
| Model providers with fallbacks across free tiers | `engine/providers.py` |
| Selective quotation: the sentence cut before its qualification, by string comparison | `engine/truncation.py` |
| The opposing-counsel memo, written from the verdict object and nothing else | `engine/memo.py` |
| The annotated brief and the verification report | `engine/report.py` |
| The drafting gate: bind a proposition to an authority, or refuse to | `engine/authority.py` |
| Draft assembly in Indian written-submission format, with a DOCX export and a verification appendix | `drafting/` |
| Self-attack on the assembled draft, from the citation graph and bench strengths, with no model | `drafting/attack.py` |
| One page for all of it: verdict board, judgment viewer, authority search, drafting workspace | `web/` |
| Verdict assembly and the grading rubric | `engine/verdict.py` |
| The engine as a LangGraph state graph | `engine/graph.py` |
| The evaluation harness: plant known failures, score all four directions | `evaluation/` |
| Repairs to stored text when extraction is corrected after the fact | `ingest/repair.py` |
| Rhetorical role for every paragraph — facts, issues, argument, ratio, disposition — by cue, never by model | `ingest/roles.py` |
| The paragraph vectors in a local Chroma database, for anything outside this process that wants to query them | `engine/chroma_store.py` |

Measured on the real corpus, which is now **the whole Supreme Court, 1950 to 2025**:

| | |
|---|---|
| Judgments (1950-2025) | 38,032 |
| Judgments with full text | 38,005 (99.93%) |
| Paragraphs indexed | 707,647 |
| Paragraphs with a vector | 668,272 |
| Date range | 1950-03-14 to 2025-12-12 |

The corpus grew from 9,429 judgments to 38,032 on 9-10 September, and the reason is worth saying
because it was a mistake rather than a milestone: **the open-data bucket always held 1950 onwards.**
A comment in `ingest/corpus.py` asserted its metadata started in 2013, and it did not — 2013 was the
year range this repository happened to have been built on. Ingesting the rest was a flag, not a
feature. The lesson is the one the engine already applies to citations: an assertion nobody checked
against the source reads exactly like a fact.

- Importing metadata for all seventy-six years takes about fifteen minutes.
- The text pass runs at about 65 judgments a minute on eight workers, so a full-corpus build is near
  ten hours; it is resumable, so the estimate survives the machine it is measured on. Of 38,032
  judgments, **27** hold no text, and each is a PDF missing from the source itself — confirmed by
  re-running `ingest bulk-text --retry` against them.
- Negative treatment is rare, as it should be: on the 2013-2025 corpus the citator found 25 across
  9,429 judgments. Each was read against the judgment that produced it over four rounds of auditing,
  and every false positive that reading found is pinned as a test. The precision bar is asymmetric: a
  missed overruling costs an advocate nothing they did not already lack, while a false one has them
  drop a binding authority.

**Which numbers on this page are from which corpus.** The index, the alias inference and the citator
were rebuilt over the full corpus, but the counts they produced have not been published, and the
recall tables above were measured on the 2013-2025 corpus and have **not** been re-run against
1950-2025. They are reported as they were measured rather than quietly reattached to a corpus four
times the size. What is measured on the full corpus is the retrieval comparison below.
- `orderorder treatment INSC:2014:53` reports Pune Municipal Corporation as **overruled**, by Indore
  Development Authority v Manoharlal (five judges, 2020) at its paragraph 362, among 118 judgments
  citing it — and shows two later two-judge benches whose words claim to overrule it downgraded to
  doubt, because they could not.

A judgment is wounded not only by what was said about it but by what happened to the cases it stood
on, and the Constitution Bench said exactly that: "all other decisions in which Pune Municipal Corpn.
has been followed, are also overruled." So a judgment that relied on or followed a case since
overruled is reported as **undermined**, with the chain named: Indore Development Authority v
Shailendra (2018) relied on Pune Municipal, which was overruled in 2020. Only reliance carries the
wound, only judgments decided before the overruling, and only one hop — and the report says in terms
that this is an inference from the citation graph rather than a holding of any court.

**The court, and the facts.** Two checks close the taxonomy. A brief that calls a two-judge decision
"a Constitution Bench", or attributes a High Court judgment to the Supreme Court, is claiming an
authority binds when it does not; the record settles that without reading the judgment, and only
overstatement is a finding. And `verify --facts` compares the facts the cited case turned on with the
facts of the matter now before the court — the one question the judgment cannot answer by itself,
since the present matter is not in it. Calling an authority inapplicable is the strong claim there, so
the model must quote the judgment's own statement of the fact said to distinguish it, and a fact that
cannot be found in the text is not recorded as distinguishing.

**What the citator cannot see.** A judgment the corpus does not hold — which was 56% of the citations
these judgments make when the corpus stopped at 2013, and is now much less, though by how much is
unmeasured. And a judgment whose text arrived truncated: Vijay Latka (2016) is held as nine
paragraphs, so the authority it rests on is not in the text at all and nothing can be inferred about
it.
- Fetching and parsing a judgment's official PDF takes two to three seconds; a 51-page judgment
  segments into 118 paragraphs.
- A citation to a paragraph the judgment does not have is reported as such, which is the Delhi High
  Court failure of September 2025.
- Across eleven ingested judgments, the voice rules attribute 8 to 15 per cent of paragraphs to
  someone other than the deciding court — counsel, the court below, or a quoted precedent — and the
  disposition paragraph is found in all eleven. On the demo judgment, a brief pinpointing paragraph 3
  is told that the paragraph is the appellant's advocate speaking, with the cue quoted, and no API key
  is involved.

Two things the corpus does not tell you, which the PDF does. The metadata's judge column names only
the **presiding** judge, so bench strength arrives under-counted; the coram printed on the judgment is
read instead, because bench strength decides which precedents bind which. And the Reports open with an
**editorial headnote**, which is the publisher's summary rather than the court's words; it is stored
separately and never used as the text a pinpoint resolves against.

**The reporter's own words are not the court's.** The Reports open with an editorial headnote and
close with the editors' sign-off — the disposition restated, the name of whoever wrote the headnote —
which extraction ran together with the court's last paragraph in two thirds of the corpus. A quote
verified against that would be reported as the court's, so both ends are now cut, and 424,000
characters of publisher's text came out of judgments already stored.

### Not built yet

A strong encoder, *run*. The seam is built, the measurement is waiting, and it is no longer gated on
hardware: `orderorder embed --api` encodes through any OpenAI-compatible `/v1/embeddings` endpoint, and
the corpus is about 263-300M tokens — roughly **$3** at BGE-M3 prices. Everything this README says
about dense retrieval is therefore a statement about a *static 8M* model, and should not be read as a
statement about dense retrieval. The contrary search is a *lead* generator and not yet a finding: `orderorder contrary` establishes
that a court wrote a sentence on the subject with the opposite polarity, and only a reading can say
whether that sentence denies the proposition or merely confines the rule to other facts. The second
reading is written and is off by default until the number beside it is measured on something other
than a laptop. Nothing yet runs the contrary search over an assembled draft, which is ten lines and
belongs beside the rest of the self-attack. OCR, so a brief filed as a scan is read rather than
reported as having no text layer. See
[docs/ROADMAP.md](docs/ROADMAP.md).

## Status

Documents complete. Both directions of the engine work end to end on the whole corpus, all twelve
failure modes are implemented, and both directions are measured on a held-out set. Started
4 September 2026.

## Attribution

Judgment data from the [Indian Supreme Court Judgments](https://registry.opendata.aws/indian-supreme-court-judgments/) and [Indian High Court Judgments](https://registry.opendata.aws/indian-high-court-judgments/) datasets on AWS Open Data (CC-BY-4.0). Lookups powered by [IKanoon](https://api.indiankanoon.org/) where indicated.

*OrderOrder is a research aid. The advocate remains responsible for every citation filed.*
