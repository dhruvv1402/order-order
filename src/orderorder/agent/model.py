"""Which model the agent thinks with, and what happens when the first choice is not there.

Strands reaches Bedrock by default and so does this, for a reason beyond the hackathon's AgentCore
bonus: the corpus already comes off AWS Open Data and the deployment in docs/DEPLOYMENT.md runs on
AWS, so one credential covers both the data and the model.

It falls back, though, and that is deliberate. `engine/providers.py` keeps a chain of providers
because a free-tier key runs out mid-run and the engine still has to answer; a demo recorded against
an expired AWS key is the same failure with worse timing. So the agent reuses that chain: when no AWS
credential resolves, the first configured provider on it answers instead, through LiteLLM.

The fallback is never silent. `ModelChoice.reason` says which model answered and why it was chosen,
`orderorder agent --which` prints it, and /api/agent returns it alongside every answer, because "the
agent got worse today" and "the agent has been on the 4B local model since Tuesday" are
indistinguishable from the outside.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from orderorder.config import Settings, get_settings
from orderorder.engine.providers import ProviderSpec, available_specs, parse_spec

# How engine/providers.py spells a provider, and how LiteLLM spells the same one. Only the providers
# that chain can actually hold are listed; anything else is passed through as written, which gets the
# right behaviour from LiteLLM for a prefix it knows and the right error for one it does not.
LITELLM_PREFIX = {
    "google_genai": "gemini",
    "groq": "groq",
    "cerebras": "cerebras",
    "openai": "openai",
    "anthropic": "anthropic",
    "ollama": "ollama",
}

# Any one of these means a credential is configured without boto3 resolving anything. The list is
# deliberately not exhaustive: a container running under a task role sets none of them, which is what
# the chain lookup in `bedrock_ready` is for.
AWS_CREDENTIAL_ENV = (
    "AWS_ACCESS_KEY_ID",
    "AWS_PROFILE",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
)

BEDROCK_PREFIX = "bedrock:"

# Every model call in this project is made at temperature zero, and the agent is not the place to
# start sampling. Its job is to choose which of eight checks answers the question asked; a second run
# that chose differently would make two identical questions produce two different audits.
TEMPERATURE = 0.0


class NoModelConfigured(RuntimeError):
    """Nothing the agent could think with. Raised where the agent is built, not at the first question."""


@dataclass(frozen=True)
class ModelChoice:
    """The model the agent will use, and the account of how it came to be that one."""

    model: Any  # a Strands model provider, typed loosely so that importing this costs no Strands import
    model_id: str
    via: str  # bedrock | litellm
    reason: str

    def describe(self) -> str:
        return f"{self.model_id} via {self.via} ({self.reason})"

    def as_json(self) -> dict:
        return {"model": self.model_id, "via": self.via, "reason": self.reason}


def bedrock_ready() -> bool:
    """Whether a Bedrock call could be signed right now.

    Asked of the whole credential chain rather than of one variable, because the three ways this
    project actually runs differ: a laptop with keys in `.env`, a container with a task role and no
    variables at all, and CI with a web identity token. Checking `AWS_ACCESS_KEY_ID` alone would send
    the deployed container down the fallback path while a perfectly good role sat unused.

    Worst case this costs about a second: with no variables set and no config file, botocore's last
    resort is the instance metadata service, whose default is a one-second timeout and one attempt.
    """
    if any(os.environ.get(name) for name in AWS_CREDENTIAL_ENV):
        return True
    try:
        import botocore.session

        return botocore.session.get_session().get_credentials() is not None
    except Exception:
        # An unreadable profile, a malformed config file, botocore absent. All of them mean the same
        # thing here -- Bedrock is not available -- and none of them should stop the agent starting.
        return False


def litellm_model_id(spec: ProviderSpec) -> str:
    """A provider string in the engine's spelling, as LiteLLM spells it.

    The model half is left exactly as written. `groq:openai/gpt-oss-120b` becomes
    `groq/openai/gpt-oss-120b`, slash and all: that slash belongs to the model's name on Groq, and
    tidying it up here would ask Groq for a model it does not have.
    """
    return f"{LITELLM_PREFIX.get(spec.provider, spec.provider)}/{spec.model}"


def _bedrock_region(settings: Settings) -> str:
    """The region to call Bedrock in.

    `BEDROCK_REGION` wins when it is set, because somebody who named it meant it. Otherwise the
    standard AWS variables are read before the default, so a box already configured for one region
    does not quietly have its model calls sent to another.
    """
    if "bedrock_region" in settings.model_fields_set:
        return settings.bedrock_region
    return os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or settings.bedrock_region


def _bedrock(model_id: str, settings: Settings, *, reason: str) -> ModelChoice:
    from strands.models import BedrockModel

    region = _bedrock_region(settings)
    model = BedrockModel(model_id=model_id, region_name=region, temperature=TEMPERATURE)
    return ModelChoice(model, f"{model_id} [{region}]", "bedrock", reason)


def _litellm(spec: ProviderSpec, *, reason: str) -> ModelChoice:
    from strands.models.litellm import LiteLLMModel

    model_id = litellm_model_id(spec)
    # The key is handed over explicitly rather than left to LiteLLM's own environment lookup, because
    # `ProviderSpec` supports a per-entry key variable (`groq:...#GROQ_API_KEY_2`, one provider on two
    # accounts) and LiteLLM only ever reads the provider's default name. Without this, the second
    # account's rung of the chain would quietly spend the first account's key.
    client_args: dict[str, Any] = {}
    env = spec.env_var
    if env and os.environ.get(env):
        client_args["api_key"] = os.environ[env]
    model = LiteLLMModel(
        client_args=client_args or None,
        model_id=model_id,
        params={"temperature": TEMPERATURE},
    )
    return ModelChoice(model, model_id, "litellm", reason)


def choose_model() -> ModelChoice:
    """Pick the model, in the order a deployment would want it picked.

    Raises `NoModelConfigured` rather than returning something that cannot answer. An agent built
    around a model that is not there fails at the first question, which is the middle of a demo; this
    fails where the agent is constructed, with a sentence naming what to set.
    """
    settings = get_settings()

    forced = (settings.agent_model or "").strip()
    if forced:
        if forced.startswith(BEDROCK_PREFIX):
            model_id = forced[len(BEDROCK_PREFIX) :].strip()
            if not model_id:
                raise NoModelConfigured(f"AGENT_MODEL is {forced!r}, which names no model after the colon")
            return _bedrock(model_id, settings, reason="AGENT_MODEL names it")
        spec = parse_spec(forced)
        if spec is None:
            raise NoModelConfigured(
                f"AGENT_MODEL is {forced!r}, which is neither 'bedrock:<model id>' nor a "
                f"'provider:model' string such as 'groq:openai/gpt-oss-120b'"
            )
        return _litellm(spec, reason="AGENT_MODEL names it")

    if bedrock_ready():
        return _bedrock(settings.bedrock_model, settings, reason="an AWS credential resolved")

    specs = available_specs()
    if specs:
        return _litellm(
            specs[0],
            reason="no AWS credential resolved, so the first configured provider answers instead",
        )

    raise NoModelConfigured(
        "the agent has no model. Configure AWS credentials for Bedrock (AWS_ACCESS_KEY_ID and "
        "AWS_SECRET_ACCESS_KEY, or AWS_PROFILE), or set a key for one of the providers in "
        "LLM_PRIMARY / LLM_FALLBACKS (GOOGLE_API_KEY, GROQ_API_KEY, CEREBRAS_API_KEY), or run Ollama "
        "locally. AGENT_MODEL overrides the choice entirely."
    )
