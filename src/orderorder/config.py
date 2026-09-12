"""Runtime settings. Everything comes from environment variables or a .env file; see .env.example."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Put .env into the process environment, not only into Settings below.
#
# Provider keys are read from os.environ — by `providers.is_available`, and by the LangChain client
# itself, which never sees this class. Without this line a key written into .env exactly as
# .env.example instructs is invisible: `doctor` reports "no key", the engine skips the model, and
# every extent-of-support check reports "not assessed" as though nothing were configured. Silent, and
# indistinguishable from having no key at all.
#
# Real environment variables win: `override=False` means an exported key beats the file, which is what
# a shell session and a container both expect.
#
# The search starts at the working directory, not at this file. Searching from the module would find
# the repository's own .env when running from a source checkout and nothing at all once the package is
# installed elsewhere — so it would work during development and quietly stop working for a user who
# installed it, which is the worst of both.
load_dotenv(find_dotenv(usecwd=True), override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Storage. The corpus, caches and database volume live here; it grows to a few gigabytes, so
    # point ORDERORDER_DATA_DIR at somewhere with room rather than accepting the default.
    orderorder_data_dir: Path = Field(default=Path("data"))
    database_url: str = ""

    # Language models as LangChain provider strings ("provider:model").
    #
    # `gemini-2.5-flash` was the default here until it was retired: Google now answers 404 for it on
    # new keys, saying to use `gemini-3.6-flash` instead. That is worth knowing about because of how
    # it failed rather than that it failed. Every model call in this engine is wrapped in a caller
    # that turns an error into "not assessed", which is the correct behaviour and meant a retired
    # model looked exactly like a corpus with nothing to say: an evaluation run against it scored
    # 100% abstention and mode 4 at 0/14, and read as a result. Pin a model, and check `doctor
    # --probe` when a report comes back emptier than the last one.
    llm_primary: str = "google_genai:gemini-3.6-flash"
    llm_fallbacks: str = "groq:openai/gpt-oss-120b,cerebras:gpt-oss-120b,ollama:qwen3.5:4b"
    llm_long_context: str = "google_genai:gemini-3.6-flash"
    llm_sensitive: str = "groq:openai/gpt-oss-120b"
    # An OpenAI-compatible endpoint to send `openai:` provider strings to, instead of OpenAI itself.
    # This is one setting for three cases the project actually has: a router or gateway, a self-hosted
    # SGLang or vLLM server (the production plan in docs/TECH_STACK.md section 5), and any of the
    # providers that speak the OpenAI protocol. Empty means talk to OpenAI.
    llm_base_url: str = ""

    # The agent layer (src/orderorder/agent). Strands reaches Bedrock by default and so does this:
    # the corpus already comes off AWS Open Data and the deployment in docs/DEPLOYMENT.md runs there,
    # so one credential covers both the data and the model. When no AWS credential resolves the agent
    # falls back to whichever provider above is configured, through LiteLLM -- a demo recorded against
    # a dead key is a demo that does not exist. `agent_model` forces one and skips the choice: either
    # "bedrock:<model id>" or a provider string in the same spelling as `llm_primary`.
    agent_model: str = ""
    bedrock_model: str = "global.anthropic.claude-sonnet-4-6"
    bedrock_region: str = "us-west-2"

    # Indian Kanoon API.
    indiankanoon_token: str | None = None
    indiankanoon_daily_quota: int = 200

    # A hosted embedding endpoint (OpenAI-compatible /v1/embeddings): Bitdeer's BAAI/bge-m3, a
    # self-hosted Qwen3-Embedding, anything that speaks the protocol. Empty means encode locally
    # with the static model, which is the measured baseline the API is an upgrade from.
    embeddings_base_url: str = ""
    embeddings_api_key: str = ""

    # Local models (later build phases).
    embeddings_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Tracing.
    langsmith_tracing: bool = False

    @property
    def data_dir(self) -> Path:
        self.orderorder_data_dir.mkdir(parents=True, exist_ok=True)
        return self.orderorder_data_dir

    @property
    def corpus_dir(self) -> Path:
        d = self.data_dir / "corpus"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def db_url(self) -> str:
        """SQLite inside the data directory unless DATABASE_URL points at Postgres."""
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'orderorder.sqlite3').as_posix()}"

    @property
    def is_postgres(self) -> bool:
        return self.db_url.startswith("postgresql")

    @property
    def fallback_models(self) -> list[str]:
        return [m.strip() for m in self.llm_fallbacks.split(",") if m.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
