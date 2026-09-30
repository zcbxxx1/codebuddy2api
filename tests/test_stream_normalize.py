"""SSE 规整：后端非标准 chunk（空字符串/空数组字段）必须被规整成标准 OpenAI 形状。

背景：后端每个分片都固定带 content:"" / reasoning_content:"" / tool_calls:[] /
finish_reason:"" / usage:null，部分客户端按「字段是否存在」判断类型，
导致思考与正文串台、切出大量空思考块（Cherry Studio 已复现）。
"""

import asyncio
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.converter import _normalize_chunk  # noqa: E402


def _chunk(delta: dict, finish_reason="", usage=None) -> dict:
    return {
        "id": "cmb-1",
        "model": "deepseek-v4.1-flash",
        "object": "chat.completion.chunk",
        "created": 1,
        "choices": [
            {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}
        ],
        "usage": usage,
    }


BACKEND_DELTA_KEYS = {
    "content": "",
    "reasoning_content": "",
    "function_call": None,
    "refusal": "",
    "tool_calls": [],
    "extra_fields": None,
}


def test_empty_fields_are_dropped():
    out = _normalize_chunk(_chunk(dict(BACKEND_DELTA_KEYS, reasoning_content="The")))
    assert out["choices"][0]["delta"] == {"reasoning_content": "The"}


def test_content_chunk_keeps_only_content():
    out = _normalize_chunk(_chunk(dict(BACKEND_DELTA_KEYS, content="你好")))
    assert out["choices"][0]["delta"] == {"content": "你好"}


def test_finish_reason_empty_string_becomes_none():
    out = _normalize_chunk(_chunk(dict(BACKEND_DELTA_KEYS), finish_reason=""))
    assert out["choices"][0]["finish_reason"] is None
    out = _normalize_chunk(_chunk(dict(BACKEND_DELTA_KEYS), finish_reason="stop"))
    assert out["choices"][0]["finish_reason"] == "stop"


def test_null_usage_dropped_but_real_usage_kept():
    assert "usage" not in _normalize_chunk(_chunk(dict(BACKEND_DELTA_KEYS)))
    usage = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    assert _normalize_chunk(_chunk(dict(BACKEND_DELTA_KEYS), usage=usage))["usage"] == usage


def test_role_and_tool_calls_preserved():
    out = _normalize_chunk(_chunk(dict(BACKEND_DELTA_KEYS, role="assistant")))
    assert out["choices"][0]["delta"] == {"role": "assistant"}

    tc = [{"index": 0, "id": "call_1", "function": {"name": "fs_read", "arguments": ""}}]
    out = _normalize_chunk(_chunk(dict(BACKEND_DELTA_KEYS, tool_calls=tc)))
    assert out["choices"][0]["delta"]["tool_calls"] == tc


def test_empty_function_call_placeholder_dropped():
    """后端用 {"name":"","arguments":""} 占位废弃的 function_call，需一并省略。"""
    out = _normalize_chunk(
        _chunk(dict(BACKEND_DELTA_KEYS, function_call={"name": "", "arguments": ""}))
    )
    assert "function_call" not in out["choices"][0]["delta"]

    real = {"name": "fs_read", "arguments": '{"path":"a"}'}
    out = _normalize_chunk(_chunk(dict(BACKEND_DELTA_KEYS, function_call=real)))
    assert out["choices"][0]["delta"]["function_call"] == real


def test_original_object_not_mutated():
    original = _chunk(dict(BACKEND_DELTA_KEYS, reasoning_content="x"))
    snapshot = json.dumps(original, sort_keys=True)
    _normalize_chunk(original)
    assert json.dumps(original, sort_keys=True) == snapshot


def _upstream_sse() -> bytes:
    """还原后端真实分片形状：先思考后正文，每个分片都带全套空字段。"""
    parts = []
    for r in ("The", " user", " is", " asking"):
        parts.append(json.dumps(_chunk(dict(BACKEND_DELTA_KEYS, reasoning_content=r)),
                                ensure_ascii=False, separators=(",", ":")))
    for c in ("你好", "，世界"):
        parts.append(json.dumps(_chunk(dict(BACKEND_DELTA_KEYS, content=c)),
                                ensure_ascii=False, separators=(",", ":")))
    parts.append(json.dumps(_chunk(dict(BACKEND_DELTA_KEYS, role="assistant"),
                                   finish_reason="stop",
                                   usage={"prompt_tokens": 5, "completion_tokens": 2,
                                          "total_tokens": 7}),
                             ensure_ascii=False, separators=(",", ":")))
    body = "".join(f"data: {p}\n\n" for p in parts) + "data: [DONE]\n\n"
    return body.encode()


def _mocked_converter(monkeypatch):
    """把转换器指向 mock 上游，返回真实 httpx.AsyncClient 供 ASGITransport 使用。"""
    from core import converter

    real_client = httpx.AsyncClient

    def upstream(request):
        return httpx.Response(200, content=_upstream_sse(),
                              headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(
        converter.httpx, "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(upstream), **kw),
    )
    monkeypatch.setattr(converter, "_check_auth", lambda *a: None)
    monkeypatch.setattr(converter, "_cred", lambda: type("Credential", (), {
        "get_headers": lambda self: {}, "backend": lambda self: "https://example.com"})())
    monkeypatch.setattr(converter, "_log", lambda *a: None)
    return real_client


def test_streaming_forwards_normalized_chunks(monkeypatch):
    from core import converter

    real_client = _mocked_converter(monkeypatch)

    async def run():
        async with real_client(transport=httpx.ASGITransport(app=converter.app),
                               base_url="http://test") as client:
            r = await client.post("/v1/chat/completions", json={
                "model": "deepseek-v4.1-flash", "stream": True,
                "messages": [{"role": "system", "content": "hi"},
                             {"role": "user", "content": "hi"}]})
            assert r.status_code == 200
            return r.text

    text = asyncio.run(run())
    deltas = [json.loads(l[6:])["choices"][0]["delta"]
              for l in text.splitlines() if l.startswith("data: ") and l[6:].strip() != "[DONE]"]

    # 思考阶段只带 reasoning_content，正文阶段只带 content —— 不串台
    thinking = [d["reasoning_content"] for d in deltas if "reasoning_content" in d]
    contents = [d["content"] for d in deltas if "content" in d]
    assert "".join(thinking) == "The user is asking"
    assert "".join(contents) == "你好，世界"

    # 任何分片都不得再出现空字段
    for d in deltas:
        assert d.get("content") != ""
        assert d.get("reasoning_content") != ""
        assert "tool_calls" not in d
        assert "function_call" not in d
        assert "extra_fields" not in d
        assert "refusal" not in d

    # finish_reason 归一为 "stop"，且 [DONE] 正常收尾
    finishes = [json.loads(l[6:])["choices"][0]["finish_reason"]
                for l in text.splitlines() if l.startswith("data: ") and l[6:].strip() != "[DONE]"]
    assert finishes[-1] == "stop"
    assert text.rstrip().endswith("[DONE]")


def test_nonstream_response_keeps_reasoning_content(monkeypatch):
    """非流式聚合必须保留 reasoning_content，否则客户端看不到思考内容。"""
    from core import converter

    real_client = _mocked_converter(monkeypatch)

    async def run():
        async with real_client(transport=httpx.ASGITransport(app=converter.app),
                               base_url="http://test") as client:
            r = await client.post("/v1/chat/completions", json={
                "model": "deepseek-v4.1-flash", "stream": False,
                "messages": [{"role": "system", "content": "hi"},
                             {"role": "user", "content": "hi"}]})
            assert r.status_code == 200
            return r.json()

    msg = asyncio.run(run())["choices"][0]["message"]
    assert msg["content"] == "你好，世界"
    assert msg["reasoning_content"] == "The user is asking"


def test_nonstream_omits_reasoning_when_absent(monkeypatch):
    """没有思考内容时不应凭空出现 reasoning_content 字段。"""
    from core import converter

    real_client = httpx.AsyncClient

    def upstream(request):
        body = (b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
                b'data: [DONE]\n\n')
        return httpx.Response(200, content=body)

    monkeypatch.setattr(
        converter.httpx, "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(upstream), **kw),
    )
    monkeypatch.setattr(converter, "_check_auth", lambda *a: None)
    monkeypatch.setattr(converter, "_cred", lambda: type("Credential", (), {
        "get_headers": lambda self: {}, "backend": lambda self: "https://example.com"})())
    monkeypatch.setattr(converter, "_log", lambda *a: None)

    async def run():
        async with real_client(transport=httpx.ASGITransport(app=converter.app),
                               base_url="http://test") as client:
            r = await client.post("/v1/chat/completions", json={
                "model": "m", "stream": False,
                "messages": [{"role": "user", "content": "hi"}]})
            return r.json()

    msg = asyncio.run(run())["choices"][0]["message"]
    assert msg["content"] == "ok"
    assert "reasoning_content" not in msg
