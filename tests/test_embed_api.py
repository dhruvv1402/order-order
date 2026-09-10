"""The hosted embedding half: any OpenAI-compatible /v1/embeddings endpoint, configured or not.

What these tests pin down without touching a network: batches are sized and unit-normalized, a
refused request (402/401) fails loudly with the endpoint's own words, the query encoder follows
whatever wrote the index, and the streaming build resumes from its watermark -- because a run that
costs real money must not pay twice. The transport is mocked; the batching, the retry arithmetic
and the JSON parsing are the real code.
"""

from __future__ import annotations

import httpx
import numpy as np
import pytest

from orderorder.engine import embeddings

DIM = 8


def _handler_factory(calls: list, fail_after: int | None = None):
    rng = np.random.default_rng(7)

    def handler(request: httpx.Request) -> httpx.Response:
        import json as jsonlib

        body = jsonlib.loads(request.content)
        calls.append(list(body["input"]))
        if fail_after is not None and len(calls) > fail_after:
            return httpx.Response(
                402, json={"error": {"message": "insufficient balance"}}
            )
        vectors = rng.normal(size=(len(body["input"]), DIM))
        return httpx.Response(200, json={"data": [{"embedding": v.tolist()} for v in vectors]})

    return handler


@pytest.fixture
def api_settings(monkeypatch):
    class S:
        embeddings_base_url = "https://embed.example/v1"
        embeddings_api_key = "test-key"
        embeddings_model = "BAAI/bge-m3"

    monkeypatch.setattr(embeddings, "get_settings", lambda: S())
    return S()


@pytest.fixture
def stub_transport(monkeypatch):
    """Real httpx.Client, mocked transport -- the wire format is exercised, the network is not."""
    calls: list = []
    state = {"fail_after": None}

    def install(fail_after: int | None = None) -> list:
        state["fail_after"] = fail_after
        handler = _handler_factory(calls, fail_after)
        real_client = httpx.Client

        def client_factory(*args, timeout=None, **kwargs):
            # Our calls carry no transport; anyone else in the process (HF hub downloads) does, and
            # must keep the real one.
            if "transport" in kwargs:
                return real_client(*args, timeout=timeout, **kwargs)
            return real_client(*args, timeout=timeout, transport=httpx.MockTransport(handler))

        monkeypatch.setattr(embeddings.httpx, "Client", client_factory)
        return calls

    return install


def test_encode_api_batches_normalizes_and_records_size(api_settings, stub_transport) -> None:
    calls = stub_transport()
    vectors = embeddings.encode_api([f"paragraph {i}" for i in range(100)], model="BAAI/bge-m3")

    assert len(calls) == 2  # 100 texts / batch 64
    assert len(calls[0]) == 64 and len(calls[1]) == 36
    assert vectors.shape == (100, DIM)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_a_refused_endpoint_fails_with_its_own_words(api_settings, stub_transport) -> None:
    stub_transport(fail_after=0)
    with pytest.raises(RuntimeError, match="insufficient balance"):
        embeddings.encode_api(["probe"], model="BAAI/bge-m3")


def test_queries_follow_whatever_encoded_the_index(api_settings, stub_transport) -> None:
    """An index written by the endpoint gets endpoint queries; a local index stays local."""
    calls = stub_transport()
    vectors = embeddings.encode_for_index(["query"], index_model="BAAI/bge-m3")
    assert vectors.shape == (1, DIM)
    assert len(calls) == 1  # went to the endpoint

    # The local static model downloads on first use; skip if it is not cached.
    try:
        local = embeddings.encode_for_index(["query"], index_model="minishlab/potion-base-8M")
    except Exception:
        pytest.skip("local static model not cached")
    assert len(calls) == 1  # did NOT go to the endpoint
    assert local.shape[1] != DIM


def test_build_api_resumes_from_the_watermark(tmp_path, api_settings, stub_transport, monkeypatch) -> None:
    """A run killed at row 3 of 5 must re-embed only the missing window."""
    calls = stub_transport()
    store = embeddings.VectorStore(directory=tmp_path / "vectors")
    store.directory.mkdir(parents=True, exist_ok=True)
    rows = [(f"p{i}", f"body {i}") for i in range(1, 6)]
    (tmp_path / "vectors" / "embed.progress.json").write_text(
        '{"model": "BAAI/bge-m3", "count": 5, "done": 3}'
    )
    np.save(store.vectors_path, np.zeros((5, DIM), dtype=np.float16))
    monkeypatch.setattr(embeddings, "paragraphs_to_embed", lambda session: rows)

    written = embeddings.build_api(session=None, store=store)

    assert written == 5
    fetched = sum(len(c) for c in calls)
    assert fetched == 2  # rows 4 and 5 only -- the watermark held
    assert store.read_index()["model"] == "BAAI/bge-m3"
    assert not (tmp_path / "vectors" / "embed.progress.json").exists()


def test_a_trial_limit_embeds_only_the_first_n_rows(tmp_path, api_settings, stub_transport, monkeypatch) -> None:
    """--limit 3 of a five-row corpus: three rows fetched, an index of three, the store stands alone."""
    calls = stub_transport()
    store = embeddings.VectorStore(directory=tmp_path / "vectors-trial")
    rows = [(f"p{i}", f"body {i}") for i in range(1, 6)]
    monkeypatch.setattr(embeddings, "paragraphs_to_embed", lambda session: rows)

    written = embeddings.build_api(session=None, store=store, limit=3)

    assert written == 3
    assert sum(len(c) for c in calls) == 3
    assert store.read_index()["count"] == 3
    assert store.read_index()["paragraph_ids"] == ["p1", "p2", "p3"]
