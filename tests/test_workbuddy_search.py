"""workbuddy_search / search_gateway 的单元测试（离线，不触网）。"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.workbuddy_search import (  # noqa: E402
    SearchError,
    SearchOutcome,
    WorkBuddySession,
    _post_json,
    build_search_query,
    load_session,
    search,
    to_anthropic_messages_response,
)


# ---------------------------------------------------------------------------
# build_search_query
# ---------------------------------------------------------------------------


class TestBuildSearchQuery:
    def test_strips_plugin_prefix(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Perform a web search for the query: rust async"}
                ],
            }
        ]
        assert build_search_query(msgs) == "rust async"

    def test_plain_string_content(self):
        assert build_search_query([{"role": "user", "content": "hello world"}]) == "hello world"

    def test_keeps_query_without_prefix(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "直接查询"}]}]
        assert build_search_query(msgs) == "直接查询"

    def test_ignores_non_text_blocks(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {}},
                    {"type": "text", "text": "Perform a web search for the query: abc"},
                ],
            }
        ]
        assert build_search_query(msgs) == "abc"

    def test_empty_messages(self):
        assert build_search_query([]) == ""
        assert build_search_query(None) == ""  # type: ignore[arg-type]

    def test_prefix_only_yields_empty(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "Perform a web search for the query: "}]}
        ]
        assert build_search_query(msgs) == ""


# ---------------------------------------------------------------------------
# to_anthropic_messages_response —— DSH 插件依赖的 wire format
# ---------------------------------------------------------------------------


class TestAnthropicResponseShape:
    def _outcome(self):
        return SearchOutcome(
            query="q",
            results=[
                {"url": "https://a.test", "title": "A", "snippet": "snip-a"},
                {"url": "https://b.test", "title": "B", "snippet": ""},
            ],
            provider=0,
            total_results=2,
            elapsed_ms=12,
        )

    def test_has_web_search_tool_result_block(self):
        """插件要求至少一个 web_search_tool_result 块，否则抛 WEB_PROVIDER_ERROR。"""
        resp = to_anthropic_messages_response(self._outcome())
        types = [b["type"] for b in resp["content"]]
        assert "web_search_tool_result" in types
        assert resp["type"] == "message"
        assert resp["role"] == "assistant"

    def test_result_items_are_web_search_result_with_url(self):
        resp = to_anthropic_messages_response(self._outcome())
        block = next(b for b in resp["content"] if b["type"] == "web_search_tool_result")
        for item in block["content"]:
            assert item["type"] == "web_search_result"
            assert item["url"]

    def test_snippet_goes_into_citations_cited_text(self):
        """关键：插件只从 text 块的 citations[].cited_text 取 snippet。"""
        resp = to_anthropic_messages_response(self._outcome())
        text = next(b for b in resp["content"] if b["type"] == "text")
        cited = {c["url"]: c["cited_text"] for c in text["citations"]}
        assert cited["https://a.test"] == "snip-a"
        # 空 snippet 不应产生 citation
        assert "https://b.test" not in cited

    def test_no_text_block_when_no_snippets(self):
        out = SearchOutcome(query="q", results=[{"url": "https://a.test", "title": "A", "snippet": ""}])
        resp = to_anthropic_messages_response(out)
        assert [b["type"] for b in resp["content"]] == ["web_search_tool_result"]

    def test_plugin_mapper_recovers_url_and_snippet(self):
        """模拟插件 mapAnthropicResponse 的逻辑，验证端到端可还原。"""
        resp = to_anthropic_messages_response(self._outcome())

        # 复刻插件实现
        snippets = {}
        for b in resp["content"]:
            if b["type"] != "text":
                continue
            for c in b.get("citations") or []:
                if c.get("url") and c.get("cited_text") and c["url"] not in snippets:
                    snippets[c["url"]] = c["cited_text"]

        sources = []
        seen = set()
        for b in resp["content"]:
            if b["type"] != "web_search_tool_result":
                continue
            for it in b.get("content") or []:
                if it["type"] != "web_search_result" or not it["url"] or it["url"] in seen:
                    continue
                seen.add(it["url"])
                src = {"url": it["url"]}
                if it.get("title"):
                    src["title"] = it["title"]
                if snippets.get(it["url"]):
                    src["snippet"] = snippets[it["url"]]
                sources.append(src)

        assert sources[0] == {"url": "https://a.test", "title": "A", "snippet": "snip-a"}
        assert sources[1] == {"url": "https://b.test", "title": "B"}


# ---------------------------------------------------------------------------
# search() —— 参数映射（mock 掉 HTTP）
# ---------------------------------------------------------------------------


class TestSearchRequest:
    def _session(self):
        return WorkBuddySession(
            access_token="tok", domain="www.workbuddy.ai",
            endpoint="https://www.workbuddy.ai", uid="u1",
        )

    def _capture(self, payload=None):
        captured = {}

        def fake_post(url, headers, body, timeout):
            captured["url"] = url
            captured["headers"] = headers
            captured["body"] = body
            return payload if payload is not None else {
                "results": [{"title": "T", "url": "https://x.test", "snippet": "S"}],
                "total_results": 1, "provider": 0, "response_time_ms": 5,
            }

        return captured, fake_post

    def test_url_and_auth_header(self):
        cap, fake = self._capture()
        with patch("core.workbuddy_search._post_json", side_effect=fake):
            search("hello", session=self._session())
        assert cap["url"] == "https://www.workbuddy.ai/agenttool/v1/search"
        assert cap["headers"]["Authorization"] == "Bearer tok"

    def test_body_defaults(self):
        cap, fake = self._capture()
        with patch("core.workbuddy_search._post_json", side_effect=fake):
            search("hello", session=self._session(), max_results=3)
        assert cap["body"] == {"query": "hello", "type": "text2text", "max_results": 3}

    def test_blocked_domains_become_negative_site_terms(self):
        """该端点没有 blocked_domains 字段 → 转成查询串里的 -site: 负向词。"""
        cap, fake = self._capture()
        with patch("core.workbuddy_search._post_json", side_effect=fake):
            search("hello", session=self._session(), blocked_domains=["a.com", "b.com"])
        assert cap["body"]["query"] == "hello (-site:a.com -site:b.com)"
        assert "blocked_domains" not in cap["body"]

    def test_allowed_domains_and_freshness_passthrough(self):
        cap, fake = self._capture()
        with patch("core.workbuddy_search._post_json", side_effect=fake):
            search("hello", session=self._session(), allowed_domains=["ok.com"], freshness="m1")
        assert cap["body"]["allowed_domains"] == ["ok.com"]
        assert cap["body"]["freshness"] == "m1"

    @pytest.mark.parametrize("bad", ["weekly", "d31", "m13", "y9", "1month"])
    def test_invalid_freshness_rejected(self, bad):
        _cap, fake = self._capture()
        with patch("core.workbuddy_search._post_json", side_effect=fake):
            with pytest.raises(SearchError, match="freshness"):
                search("hello", session=self._session(), freshness=bad)

    @pytest.mark.parametrize("ok", ["d1", "d30", "m1", "m12", "y1", "y5", "d", "m"])
    def test_valid_freshness_accepted(self, ok):
        cap, fake = self._capture()
        with patch("core.workbuddy_search._post_json", side_effect=fake):
            search("hello", session=self._session(), freshness=ok)
        assert cap["body"]["freshness"] == ok

    def test_empty_query_rejected(self):
        with pytest.raises(SearchError, match="不能为空"):
            search("   ", session=self._session())

    def test_results_without_url_dropped(self):
        payload = {"results": [{"title": "no url"}, {"url": "https://ok.test", "title": "ok"}]}
        _cap, fake = self._capture(payload)
        with patch("core.workbuddy_search._post_json", side_effect=fake):
            out = search("q", session=self._session())
        assert [r["url"] for r in out.results] == ["https://ok.test"]

    def test_max_results_clamped(self):
        cap, fake = self._capture()
        with patch("core.workbuddy_search._post_json", side_effect=fake):
            search("q", session=self._session(), max_results=9999)
        assert cap["body"]["max_results"] == 50


class TestPostJsonValidation:
    """_post_json 的响应校验——在真实层次上测（不 mock 掉被测逻辑本身）。"""

    class _FakeResp:
        status = 200

        def __init__(self, body: bytes):
            self._b = body

        def read(self) -> bytes:
            return self._b

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _post(self, payload):
        raw = json.dumps(payload).encode()
        with patch("urllib.request.urlopen", return_value=self._FakeResp(raw)):
            return _post_json("http://x/messages", {}, {}, 5)

    def test_business_error_code_raises(self):
        with pytest.raises(SearchError, match="15001"):
            self._post({"code": 15001, "msg": "rate limit"})

    def test_code_zero_is_success(self):
        assert self._post({"code": 0, "results": []})["code"] == 0

    def test_missing_code_is_success(self):
        assert "results" in self._post({"results": []})

    def test_http_401_raises_with_status(self):
        err = urllib.error.HTTPError("http://x", 401, "Unauthorized", {}, None)  # type: ignore[arg-type]

        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(SearchError, match="401"):
                _post_json("http://x/messages", {}, {}, 5)

    def test_non_json_body_raises(self):
        with patch("urllib.request.urlopen", return_value=self._FakeResp(b"<html>")):
            with pytest.raises(SearchError, match="非 JSON"):
                _post_json("http://x/messages", {}, {}, 5)

    def test_network_error_raises(self):
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("boom")):
            with pytest.raises(SearchError, match="无法连接"):
                _post_json("http://x/messages", {}, {}, 5)


# ---------------------------------------------------------------------------
# 凭据加载
# ---------------------------------------------------------------------------


class TestLoadSession:
    def test_plaintext_token(self, tmp_path):
        f = tmp_path / "workbuddy-desktop-ai.info"
        f.write_text(
            json.dumps(
                {
                    "auth": {"accessToken": "plain-tok", "domain": "www.workbuddy.ai", "expiresAt": 99999999999999},
                    "account": {"uid": "u-1"},
                }
            ),
            encoding="utf-8",
        )
        s = load_session(f)
        assert s.access_token == "plain-tok"
        assert s.endpoint == "https://www.workbuddy.ai"
        assert s.uid == "u-1"

    def test_encrypted_envelope_decrypted(self, tmp_path):
        f = tmp_path / "workbuddy-desktop-ai.info"
        f.write_text(
            json.dumps({"auth": {"accessToken": {"$wbEncrypted": 1, "envelope": "x"}}}),
            encoding="utf-8",
        )
        with patch("core.workbuddy_search.decrypt_auth_field", return_value="decrypted-tok"):
            s = load_session(f)
        assert s.access_token == "decrypted-tok"

    def test_missing_token_raises(self, tmp_path):
        f = tmp_path / "empty.info"
        f.write_text(json.dumps({"auth": {}}), encoding="utf-8")
        with pytest.raises(SearchError, match="登录态"):
            load_session(f)

    def test_no_candidates_raises(self, tmp_path):
        with pytest.raises(SearchError):
            load_session(tmp_path / "nope.info")

    def test_expired_flag(self):
        past = int(time.time() * 1000) - 10_000
        s = WorkBuddySession(access_token="t", expires_at=past)
        assert s.expired is True
        s2 = WorkBuddySession(access_token="t", expires_at=int(time.time() * 1000) + 600_000)
        assert s2.expired is False

    def test_no_expiry_means_not_expired(self):
        assert WorkBuddySession(access_token="t", expires_at=0).expired is False

    def test_headers_include_identity(self):
        s = WorkBuddySession(
            access_token="tok", domain="www.workbuddy.ai", uid="u9", enterprise_id="e9"
        )
        h = s.headers()
        assert h["Authorization"] == "Bearer tok"
        assert h["X-User-Id"] == "u9"
        assert h["X-Tenant-Id"] == "e9"
        assert h["X-Domain"] == "www.workbuddy.ai"

    def test_headers_omit_blank_identity(self):
        h = WorkBuddySession(access_token="tok").headers()
        assert "X-User-Id" not in h
        assert "X-Tenant-Id" not in h


# ---------------------------------------------------------------------------
# 搜索网关 HTTP 层
# ---------------------------------------------------------------------------


@pytest.fixture()
def gateway(tmp_path):
    """在临时端口启动网关，mock 掉真实搜索。"""
    from core import search_gateway as gw
    from http.server import ThreadingHTTPServer
    import threading

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), gw.GatewayHandler)
    httpd.max_results = 8  # type: ignore[attr-defined]
    httpd.verbose = False  # type: ignore[attr-defined]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _post(url, body):
    import urllib.request

    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


PLUGIN_BODY = {
    "model": "deepseek-v4-flash",
    "max_tokens": 4096,
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "Perform a web search for the query: test q"}]}
    ],
    "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
}


class TestGatewayHTTP:
    def test_real_plugin_payload_returns_tool_result(self, gateway):
        outcome = SearchOutcome(
            query="test q",
            results=[{"url": "https://x.test", "title": "X", "snippet": "snip"}],
            elapsed_ms=3,
        )
        with patch("core.search_gateway.search", return_value=outcome):
            status, resp = _post(f"{gateway}/messages", PLUGIN_BODY)
        assert status == 200
        types = [b["type"] for b in resp["content"]]
        assert "web_search_tool_result" in types
        assert resp["model"] == "deepseek-v4-flash"

    def test_query_prefix_stripped_before_upstream(self, gateway):
        outcome = SearchOutcome(query="test q", results=[], elapsed_ms=1)
        captured = {}

        def fake_search(q, **kw):
            captured["q"] = q
            return outcome

        with patch("core.search_gateway.search", side_effect=fake_search):
            _post(f"{gateway}/messages", PLUGIN_BODY)
        assert captured["q"] == "test q"

    def test_missing_tool_declaration_rejected(self, gateway):
        body = dict(PLUGIN_BODY, tools=[])
        status, resp = _post(f"{gateway}/messages", body)
        assert status == 400
        assert "web_search" in resp["error"]["message"]

    def test_unknown_path_404(self, gateway):
        status, _ = _post(f"{gateway}/v1/messages", PLUGIN_BODY)
        assert status == 404

    def test_bad_json_400(self, gateway):
        import urllib.request

        req = urllib.request.Request(
            f"{gateway}/messages", data=b"{not json", method="POST",
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(req, timeout=10)
        assert ei.value.code == 400

    def test_auth_failure_maps_to_401(self, gateway):
        with patch("core.search_gateway.search", side_effect=SearchError("HTTP 401 上游拒绝")):
            status, resp = _post(f"{gateway}/messages", PLUGIN_BODY)
        assert status == 401
        assert resp["error"]["type"] == "authentication_error"

    def test_other_failure_maps_to_502(self, gateway):
        with patch("core.search_gateway.search", side_effect=SearchError("网络不可达")):
            status, resp = _post(f"{gateway}/messages", PLUGIN_BODY)
        assert status == 502
        assert resp["error"]["type"] == "api_error"

    def test_health_ok(self, gateway):
        import urllib.request

        with patch("core.search_gateway.load_session") as ls:
            ls.return_value = WorkBuddySession(
                access_token="t", endpoint="https://www.workbuddy.ai", path=Path("x")
            )
            with urllib.request.urlopen(f"{gateway}/health", timeout=10) as r:
                body = json.loads(r.read().decode())
        assert body["status"] == "ok"
        assert body["endpoint"] == "https://www.workbuddy.ai"


# ---------------------------------------------------------------------------
# build_server / converter 内嵌启动
# ---------------------------------------------------------------------------


class TestBuildServer:
    def test_raises_without_credentials(self):
        """无可用登录态时构造即失败，而不是等请求进来才报错。"""
        from core import search_gateway as gw

        with patch("core.search_gateway.load_session", side_effect=SearchError("无凭据")):
            with pytest.raises(SearchError):
                gw.build_server("127.0.0.1", 0)

    def test_binds_ephemeral_port_and_sets_options(self):
        from core import search_gateway as gw

        with patch("core.search_gateway.load_session", return_value=WorkBuddySession(access_token="t")):
            httpd = gw.build_server("127.0.0.1", 0, max_results=3, verbose=True)
        try:
            assert httpd.max_results == 3  # type: ignore[attr-defined]
            assert httpd.verbose is True  # type: ignore[attr-defined]
            assert httpd.server_address[1] > 0
        finally:
            httpd.server_close()


class TestConverterIntegration:
    """converter 把搜索网关挂到主 app 上（单端口）。"""

    def test_mount_returns_prefix(self):
        from core import converter

        assert converter.mount_search_gateway(enabled=True) == "/v1/searchGateway"

    def test_mount_disabled_returns_none(self):
        from core import converter

        assert converter.mount_search_gateway(enabled=False) is None

    def test_mounted_routes_respond(self):
        """挂载后 /v1/searchGateway/{health,messages} 都必须真实可达。

        这里用 TestClient 走完整 ASGI 栈：能抓到诸如
        'Request 被当成 query 参数'（注解解析失败）这类只在请求期暴露的问题。
        """
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from core.search_gateway import make_search_router

        test_app = FastAPI()
        test_app.include_router(make_search_router())

        with patch("core.search_gateway.load_session", return_value=WorkBuddySession(access_token="t")):
            with TestClient(test_app) as client:
                # health 走真实 load_session（已 patch）
                assert client.get("/v1/searchGateway/health").status_code == 200

                # 无工具声明 → 400（而不是 422：那说明 request 注解没解析对）
                r = client.post("/v1/searchGateway/messages", json={"messages": []})
                assert r.status_code == 400, r.text

                # 真实搜索（mock 掉上游）
                outcome = SearchOutcome(
                    query="q",
                    results=[{"url": "https://x.test", "title": "X", "snippet": "s"}],
                )
                with patch("core.search_gateway.search", return_value=outcome):
                    r = client.post(
                        "/v1/searchGateway/messages",
                        json={
                            "model": "deepseek-v4-flash",
                            "messages": [{"role": "user", "content": [
                                {"type": "text", "text": "Perform a web search for the query: q"}]}],
                            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                        },
                    )
                assert r.status_code == 200, r.text
                body = r.json()
                assert any(b["type"] == "web_search_tool_result" for b in body["content"])

    def test_route_does_not_shadow_existing_endpoints(self):
        """搜索路径不能占用既有对话端点（必须挂在独立前缀下）。"""
        from core.search_gateway import SEARCH_ROUTE_PREFIX

        assert SEARCH_ROUTE_PREFIX == "/v1/searchGateway"
        # 关键：不能是 /v1 本身，否则 /v1/messages 会被插件与对话共用
        assert SEARCH_ROUTE_PREFIX != "/v1"
        assert SEARCH_ROUTE_PREFIX.startswith("/v1/")

    def test_plugin_endpoint_derivation(self):
        """插件用 `${baseURL}/messages` 拼端点，验证我们给的 baseURL 能拼对。"""
        from core.search_gateway import SEARCH_ROUTE_PREFIX

        assert SEARCH_ROUTE_PREFIX + "/messages" == "/v1/searchGateway/messages"


class TestDefaultOnStartup:
    """搜索网关默认挂载（单端口）。"""

    def _run_main(self, argv):
        """用真实 parser 跑 converter.main()，返回 mount 的调用记录。"""
        from core import converter

        with patch.object(converter.uvicorn, "run"), patch.object(
            converter, "preflight"
        ), patch.object(converter, "find_auth_file", return_value=None), patch.object(
            converter, "mount_search_gateway"
        ) as mounter, patch.object(
            converter, "_log"
        ), patch("sys.stderr"), patch.object(
            sys, "argv", ["converter", *argv]
        ):
            converter.main()
        return mounter

    def test_enabled_by_default(self):
        """不传任何搜索相关参数时，也应挂载搜索网关。"""
        mounter = self._run_main(["--skip-check"])
        assert mounter.called
        assert mounter.call_args.kwargs.get("enabled") is True

    def test_no_search_gateway_flag_disables(self):
        mounter = self._run_main(["--skip-check", "--no-search-gateway"])
        assert mounter.called
        assert mounter.call_args.kwargs.get("enabled") is False

    def test_removed_port_flag_no_longer_accepted(self):
        """旧的独立端口参数已移除（搜索现在复用主端口）。"""
        from core import converter

        with patch.object(converter, "uvicorn"), patch.object(converter, "preflight"), patch.object(
            converter, "find_auth_file", return_value=None
        ), patch.object(converter, "_log"), patch("sys.stderr"), patch.object(
            sys, "argv", ["converter", "--skip-check", "--search-gateway-port", "9100"]
        ):
            with pytest.raises(SystemExit):
                converter.main()


class TestStandaloneServerStillWorks:
    """独立调试模式保留（build_server + GatewayServer）。"""

    def test_allow_reuse_address_disabled(self):
        """http.server 默认 allow_reuse_address=1，Windows 上会让 bind 到已占用端口
        仍成功，导致两个实例并存、请求被旧进程接走。必须显式关闭。"""
        from core.search_gateway import GatewayServer

        assert GatewayServer.allow_reuse_address is False

    def test_second_bind_to_same_port_fails(self):
        from core import search_gateway as gw

        with patch("core.search_gateway.load_session", return_value=WorkBuddySession(access_token="t")):
            first = gw.build_server("127.0.0.1", 0)
        port = first.server_address[1]
        try:
            with patch("core.search_gateway.load_session", return_value=WorkBuddySession(access_token="t")):
                with pytest.raises(OSError):
                    gw.build_server("127.0.0.1", port)
        finally:
            first.server_close()

    def test_standalone_serves_same_core(self, gateway):
        """独立模式的 HTTP 行为与 FastAPI 挂载共用同一核心。"""
        with patch("core.search_gateway.search") as searcher:
            searcher.return_value = SearchOutcome(
                query="q", results=[{"url": "https://x.test", "title": "X", "snippet": "s"}]
            )
            status, resp = _post(f"{gateway}/messages", PLUGIN_BODY)
        assert status == 200
        assert any(b["type"] == "web_search_tool_result" for b in resp["content"])


class TestSharedCoreParity:
    """handle_search_request 是两条传输路径的公共核心。"""

    def _body(self):
        return {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Perform a web search for the query: q"}]}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        }

    def test_rejects_missing_tool(self):
        from core.search_gateway import SearchRequestError, handle_search_request

        with pytest.raises(SearchRequestError) as ei:
            handle_search_request({"messages": []})
        assert ei.value.status == 400

    def test_rejects_empty_query(self):
        from core.search_gateway import SearchRequestError, handle_search_request

        with pytest.raises(SearchRequestError):
            handle_search_request(
                {"messages": [{"role": "user", "content": [
                    {"type": "text", "text": "Perform a web search for the query: "}]}],
                 "tools": [{"type": "web_search_20250305"}]}
            )

    def test_maps_auth_error_to_401(self):
        from core.search_gateway import SearchRequestError, handle_search_request

        with patch("core.search_gateway.search", side_effect=SearchError("HTTP 401 拒绝")):
            with pytest.raises(SearchRequestError) as ei:
                handle_search_request(self._body())
        assert ei.value.status == 401
        assert ei.value.err_type == "authentication_error"

    def test_maps_other_error_to_502(self):
        from core.search_gateway import SearchRequestError, handle_search_request

        with patch("core.search_gateway.search", side_effect=SearchError("网络不可达")):
            with pytest.raises(SearchRequestError) as ei:
                handle_search_request(self._body())
        assert ei.value.status == 502

    def test_success_shape(self):
        from core.search_gateway import handle_search_request

        outcome = SearchOutcome(
            query="q", results=[{"url": "https://x.test", "title": "X", "snippet": "s"}]
        )
        with patch("core.search_gateway.search", return_value=outcome):
            payload = handle_search_request(self._body())
        assert payload["role"] == "assistant"
        assert any(b["type"] == "web_search_tool_result" for b in payload["content"])

    def test_has_web_search_tool_variants(self):
        from core.search_gateway import has_web_search_tool

        assert has_web_search_tool({"tools": [{"type": "web_search_20250305"}]})
        assert has_web_search_tool({"tools": [{"name": "web_search"}]})
        assert has_web_search_tool({"tools": [{"type": "web_search_preview"}]})
        assert not has_web_search_tool({"tools": [{"type": "function", "name": "grep"}]})
        assert not has_web_search_tool({})

