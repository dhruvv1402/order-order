"""The agent layer: that it offers the engine's checks, and that it cannot answer without them.

Nothing here calls a language model, and that is the point. What a model says when it reads these
tools is not testable; what is testable is everything the agent's honesty actually rests on --

  * **every check the engine has is reachable**, because a tool that was never registered is a
    question the agent will answer from memory instead, fluently and wrongly;
  * **each tool carries the description the model chooses by**, since Strands builds the schema from
    the docstring and a trimmed docstring silently becomes a mis-routed question;
  * **the tools read the database they were handed**, not the developer's corpus;
  * **the prompt still forbids what it was written to forbid**, which is one careless edit away from
    not being true;
  * **the findings keep their shape through serialisation** -- an `unchecked` treatment that arrives
    at the model as a bare "good_law" is how "nothing has ever cited this" becomes "we checked it".

The engine's own behaviour is tested in test_search, test_locator, test_citator and the rest, and is
not re-tested here. These are about the seam.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from orderorder.agent import SYSTEM_PROMPT, build_tools
from orderorder.agent import model as model_module
from orderorder.agent.model import NoModelConfigured, choose_model, litellm_model_id
from orderorder.engine.providers import ProviderSpec, parse_spec

# Every check the engine exposes, and the one a question has to land on. Written out rather than
# derived from `build_tools`, so that deleting a tool fails here instead of passing quietly.
EXPECTED = {
    "corpus_status",
    "resolve_citation",
    "check_treatment",
    "locate_paragraph",
    "verify_brief",
    "find_authority",
    "find_contrary_authority",
    "bind_proposition",
}


@pytest.fixture
def tools(corpus) -> dict:
    """The tools, bound to the one-judgment corpus, addressable by name."""
    return {t.tool_spec["name"]: t for t in build_tools(corpus)}


def _call(tool, **kwargs) -> dict:
    """Run a tool as the decorator still allows it to be run: as the plain function it also is.

    The schema Strands generates around it is asserted separately, above. What these tests are about
    is the payload, and going through the tool-use envelope would put the SDK's own dispatch in the
    middle of that without testing anything here that the spec assertion does not already cover.
    """
    return tool(**kwargs)


def test_every_check_the_engine_has_is_offered_as_a_tool(tools) -> None:
    assert set(tools) == EXPECTED


def test_each_tool_carries_the_description_the_model_chooses_by(tools) -> None:
    """Strands builds the schema from the docstring, so the docstring is load-bearing.

    A short description is not a style problem here. `check_treatment` and `find_authority` are told
    apart by these words alone, and a question about subsequent history routed to the search tool
    comes back with an answer that is about the right case and says nothing about whether it survived.
    """
    for name, tool in tools.items():
        spec = tool.tool_spec
        assert len(spec["description"]) > 250, f"{name} has lost the words the model chooses it by"
        # Strands lifts the docstring's Args section out of the description and onto each parameter,
        # so an undocumented argument is not a shorter description -- it is a parameter the model is
        # handed with no idea what belongs in it. `claimed_pinpoint` is the one that matters: filled in
        # when nothing was claimed, it invents a pinpoint for the engine to check.
        for parameter, described in spec["inputSchema"]["json"]["properties"].items():
            assert described.get("description"), f"{name} does not document {parameter}"


def test_the_prompt_keeps_the_rule_the_whole_thing_rests_on() -> None:
    """The prohibitions, spelled out, because an edit that softens one leaves every test passing.

    This is the product. An agent in front of a citation checker that will cite from memory when the
    corpus comes up empty is worse than no agent: it is the failure the engine exists to catch, served
    in the engine's own voice.
    """
    prompt = SYSTEM_PROMPT.lower()
    assert "you have no knowledge of indian case law" in prompt
    assert "never write a citation a tool did not return" in prompt
    assert "null is not false" in prompt
    assert "not assessed" in prompt
    assert "unchecked" in prompt


def test_the_tools_read_the_database_they_were_given(tools) -> None:
    """One judgment, because that is what the fixture holds. The developer's corpus has 9,429."""
    status = _call(tools["corpus_status"])
    assert status["judgments"] == 1
    assert status["judgments_with_text"] == 1
    assert status["index_built"] is True
    assert status["courts"] == {"Supreme Court of India": 1}


def test_a_citation_resolves_to_the_judgment_it_names(tools) -> None:
    found = _call(tools["resolve_citation"], citation="(2019) 4 SCC 118")["citations"]
    assert len(found) == 1
    assert found[0]["status"] == "found"
    assert found[0]["key"] == "TEST:0001:1"


def test_a_string_with_no_citation_in_it_says_so(tools) -> None:
    """Rather than returning an empty list, which a model reads as "no such case"."""
    answer = _call(tools["resolve_citation"], citation="the law on misrepresentation")
    assert answer["citations"] == []
    assert "no citation was found" in answer["note"]


def test_a_tool_given_a_citation_where_a_key_belongs_still_answers(tools) -> None:
    """The agent's most likely mistake, and the one `_judgment_from` exists to absorb."""
    by_key = _call(tools["check_treatment"], judgment="TEST:0001:1")
    by_citation = _call(tools["check_treatment"], judgment="(2019) 4 SCC 118")
    assert by_citation["judgment"] == by_key["judgment"]


def test_treatment_of_an_uncited_judgment_arrives_marked_unchecked(tools) -> None:
    """Nothing in this corpus cites the fixture's judgment, so `status` alone would mislead.

    "good_law" here means nobody has said anything about it, which is not a finding. If `unchecked`
    ever stops reaching the model, the agent starts reporting "we checked and it is sound" about
    judgments no court has ever mentioned.
    """
    report = _call(tools["check_treatment"], judgment="TEST:0001:1")
    assert report["status"] == "good_law"
    assert report["unchecked"] is True
    assert report["citing_count"] == 0


def test_find_authority_offers_the_courts_own_voice_and_not_counsels(tools) -> None:
    """Paragraph 2 is what counsel urged; paragraph 3 is what the court held.

    Paragraph 2 states the rule more broadly and matches the words of a broad proposition better,
    which is exactly why it must not come back as authority. The search drops it; this proves the
    tool did not undo that.
    """
    answer = _call(
        tools["find_authority"],
        proposition="a misrepresentation vitiates consent only where it induced the contract",
    )
    assert answer["authorities"], "the corpus holds a paragraph that says this"
    for offered in answer["authorities"]:
        assert offered["paragraph"] != "2", "counsel's argument was offered as authority"
    assert answer["authorities"][0]["paragraph"] == "3"
    assert answer["authorities"][0]["line"]


def test_locate_paragraph_reports_a_pinpoint_that_cannot_exist(tools) -> None:
    """A brief citing paragraph 9 of a judgment with four paragraphs, and the note that says so."""
    answer = _call(
        tools["locate_paragraph"],
        judgment="TEST:0001:1",
        proposition="a misrepresentation vitiates consent only where it induced the contract",
        claimed_pinpoint="9",
    )
    assert answer["pinpoint"]["claimed"] == "9"
    assert answer["pinpoint"]["status"] == "out_of_range"
    assert answer["pinpoint"]["is_problem"] is True
    assert "stops at 4" in answer["pinpoint"]["note"]


def test_the_holding_outranks_the_argument_that_shares_its_words(tools) -> None:
    """Paragraph 3 first, with paragraph 2 behind it rather than absent.

    The ranking, not the filter: `locate_paragraph` is asked which paragraph of a judgment carries a
    claim, and the answer about a judgment whose counsel argued the same point more broadly is both
    paragraphs in the right order. Dropping paragraph 2 here would hide the thing a reader needs to
    see -- that the broad version of the proposition is in the judgment, in counsel's mouth.

    A claim of paragraph 2 therefore comes back "ok" and not as a problem, which is correct and worth
    pinning down: the words really are in paragraph 2. That the paragraph is argument rather than
    holding is `find_authority`'s question and the voice check's, not the locator's.
    """
    proposition = "a misrepresentation vitiates consent only where it induced the contract"
    answer = _call(tools["locate_paragraph"], judgment="TEST:0001:1", proposition=proposition)
    assert [c["paragraph"] for c in answer["candidates"][:2]] == ["3", "2"]

    claimed = _call(
        tools["locate_paragraph"],
        judgment="TEST:0001:1",
        proposition=proposition,
        claimed_pinpoint="2",
    )
    assert claimed["pinpoint"]["status"] == "ok"
    assert claimed["pinpoint"]["is_problem"] is False


def test_an_empty_proposition_is_refused_rather_than_searched_for(tools) -> None:
    assert "error" in _call(tools["find_authority"], proposition="   ")
    assert "error" in _call(tools["verify_brief"], text="")
    assert "error" in _call(tools["bind_proposition"], proposition="")


def test_a_paragraph_is_trimmed_before_it_reaches_the_model(tools) -> None:
    """A judgment's paragraph runs to thousands of characters and the agent is asked which, not what.

    The trim has to announce itself, because a model handed a sentence that stops mid-clause will
    finish the sentence for it, and the finished half is not in the judgment.
    """
    from orderorder.agent import tools as tools_module

    trimmed = tools_module._trim("x" * (tools_module.BODY_CHARS + 500))
    assert len(trimmed) < tools_module.BODY_CHARS + 60
    assert "more characters]" in trimmed
    assert tools_module._trim("short") == "short"


# --- which model answers -------------------------------------------------------------------------


def _settings(**kwargs) -> SimpleNamespace:
    defaults = {
        "agent_model": "",
        "bedrock_model": "global.anthropic.claude-sonnet-4-6",
        "bedrock_region": "us-west-2",
        "model_fields_set": {"bedrock_region"},
    }
    return SimpleNamespace(**{**defaults, **kwargs})


def test_a_provider_string_is_spelled_the_way_litellm_spells_it() -> None:
    """The model half is left alone. The slash in `openai/gpt-oss-120b` is part of Groq's model name."""
    assert litellm_model_id(parse_spec("groq:openai/gpt-oss-120b")) == "groq/openai/gpt-oss-120b"
    assert litellm_model_id(parse_spec("google_genai:gemini-3.6-flash")) == "gemini/gemini-3.6-flash"
    assert litellm_model_id(ProviderSpec("ollama", "qwen3.5:4b")) == "ollama/qwen3.5:4b"


def test_bedrock_is_used_when_a_credential_resolves(monkeypatch) -> None:
    monkeypatch.setattr(model_module, "get_settings", _settings)
    monkeypatch.setattr(model_module, "bedrock_ready", lambda: True)
    choice = choose_model()
    assert choice.via == "bedrock"
    assert "claude-sonnet-4-6" in choice.model_id
    assert "us-west-2" in choice.model_id


def test_without_aws_the_engines_own_provider_chain_answers(monkeypatch) -> None:
    """The fallback that keeps a demo alive, and the reason it has to be visible in the answer."""
    monkeypatch.setattr(model_module, "get_settings", _settings)
    monkeypatch.setattr(model_module, "bedrock_ready", lambda: False)
    monkeypatch.setattr(model_module, "available_specs", lambda: [parse_spec("groq:openai/gpt-oss-120b")])
    choice = choose_model()
    assert choice.via == "litellm"
    assert choice.model_id == "groq/openai/gpt-oss-120b"
    assert "no AWS credential" in choice.reason


def test_agent_model_overrides_the_choice_either_way(monkeypatch) -> None:
    monkeypatch.setattr(model_module, "bedrock_ready", lambda: False)

    monkeypatch.setattr(model_module, "get_settings", lambda: _settings(agent_model="bedrock:some.model"))
    forced = choose_model()
    assert forced.via == "bedrock"
    assert forced.model_id.startswith("some.model")

    monkeypatch.setattr(model_module, "get_settings", lambda: _settings(agent_model="cerebras:gpt-oss-120b"))
    assert choose_model().model_id == "cerebras/gpt-oss-120b"


def test_no_model_fails_where_the_agent_is_built(monkeypatch) -> None:
    """Not at the first question, which is the middle of a demo. And the error names what to set."""
    monkeypatch.setattr(model_module, "get_settings", _settings)
    monkeypatch.setattr(model_module, "bedrock_ready", lambda: False)
    monkeypatch.setattr(model_module, "available_specs", lambda: [])
    with pytest.raises(NoModelConfigured) as raised:
        choose_model()
    assert "AWS_ACCESS_KEY_ID" in str(raised.value)
    assert "AGENT_MODEL" in str(raised.value)


def test_a_meaningless_agent_model_is_refused_by_name(monkeypatch) -> None:
    monkeypatch.setattr(model_module, "get_settings", lambda: _settings(agent_model="gpt-4"))
    with pytest.raises(NoModelConfigured, match="gpt-4"):
        choose_model()


# --- over HTTP -----------------------------------------------------------------------------------


def test_a_question_with_nothing_in_it_is_refused(client) -> None:
    assert client.post("/api/agent", json={"question": "   "}).status_code == 400


def test_an_unconfigured_model_is_a_503_and_not_a_crash(client, monkeypatch) -> None:
    """Nothing is broken; something is unset. The status code should say which."""
    from orderorder.web import api

    def refuse(*args, **kwargs):
        raise NoModelConfigured("no model: set AWS_ACCESS_KEY_ID or GROQ_API_KEY")

    monkeypatch.setattr(api, "build_assistant", refuse)
    response = client.post("/api/agent", json={"question": "is this good law?"})
    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]


def test_an_answer_comes_back_with_the_model_and_the_checks_that_ran(client, monkeypatch) -> None:
    """The audit trail travels with the answer, or a caller cannot tell memory from verification."""
    from orderorder.agent.model import ModelChoice
    from orderorder.web import api

    choice = ModelChoice(object(), "some/model", "litellm", "because the test said so")
    stub = SimpleNamespace(
        model=choice,
        ask=lambda question: "paragraph 2 is counsel's argument, not the holding",
        tools_used=lambda: ["resolve_citation", "locate_paragraph"],
    )
    monkeypatch.setattr(api, "build_assistant", lambda *args, **kwargs: stub)

    body = client.post("/api/agent", json={"question": "check para 2"}).json()
    assert body["answer"].startswith("paragraph 2")
    assert body["model"] == {"model": "some/model", "via": "litellm", "reason": "because the test said so"}
    assert body["tools_used"] == ["resolve_citation", "locate_paragraph"]


def test_the_status_line_says_which_model_the_agent_is_on(client) -> None:
    """Whatever this machine has configured, the health route must render and must answer the question."""
    agent = client.get("/api/health").json()["agent"]
    assert "configured" in agent
    assert agent["via"] if agent["configured"] else agent["reason"]
