"""What the process says about itself while it runs.

Until now it said nothing at all. Every degraded check in this engine records its reason on the
object it was attached to — `review_reason` on a verdict, `error` on a job — which is exactly right
for the advocate reading that verdict and useless to whoever is running the box. A provider that has
started refusing every call produces a page full of honest *not assessed* and no other trace, and
the difference between "the corpus has nothing to say about this" and "the model has been down since
Tuesday" is the difference between a result and an outage. `docs/DEPLOYMENT.md` §5 says to watch the
abstention rate for exactly this reason; a log is how you watch it without reading every verdict.

Three rules, and the third is the one that matters here.

**One logger, on stderr, quiet by default.** `ORDERORDER_LOG_LEVEL` raises or lowers it. A pilot on
one box wants its lines in the container's output where `docker logs` already looks, not in a file
somebody has to find and rotate.

**Configuring is idempotent and never done at import.** A library that installs a handler when it is
imported fights whatever the application already chose. `configure()` is called by `serve` and by the
app factory, and calling it twice changes nothing.

**No brief, no plan, no judgment text, ever.** An uploaded brief is privileged, and the single
guarantee this deployment makes about it is that it is not stored — which a log line quietly undoes.
So what is logged is what failed and why: a provider name, an exception type, a job id. Never the
prompt that was sent and never the text it was about.
"""

from __future__ import annotations

import logging
import os
import sys

LOGGER_NAME = "orderorder"

# An exception's message can carry a chunk of whatever was sent, depending on the client that raised
# it. Nothing here needs more than the first line of the reason, and a cap is cheaper than trusting
# every provider's idea of a good error string.
MAX_REASON_CHARS = 300

_configured = False


def get_logger(name: str | None = None) -> logging.Logger:
    """The logger for a module. `get_logger(__name__)` at the top of a file, as usual."""
    if not name or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(LOGGER_NAME).getChild(name.removeprefix("orderorder."))


def configure(level: str | None = None) -> None:
    """Attach one stderr handler to the `orderorder` logger. Safe to call more than once."""
    global _configured
    if _configured:
        return
    chosen = level or os.environ.get("ORDERORDER_LOG_LEVEL") or "INFO"
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(chosen.upper())
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    # Ours alone. Propagating would hand every line to the root logger too, which uvicorn has already
    # configured, and the operator would read everything twice.
    logger.propagate = False
    _configured = True


def reason(exc: BaseException) -> str:
    """An exception as one short, loggable string, in the form the rest of the codebase uses."""
    text = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    return text if len(text) <= MAX_REASON_CHARS else text[: MAX_REASON_CHARS - 1] + "…"
