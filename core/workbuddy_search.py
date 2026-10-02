#!/usr/bin/env python3
"""
workbuddy_search — 通过本机 WorkBuddy 登录态调用其搜索能力。

背景（见 DSH 搜索能力调研）：
  DSH 的 `web_search` 是**宿主侧本地工具**，由 `@deepseek-ai/dsh-web-search-deepseek`
  插件在**执行时**向一个 Anthropic Messages 兼容端点发起独立请求：

      POST {baseURL}/messages
      {
        "model": "deepseek-v4-flash", "max_tokens": 4096,
        "messages": [{"role":"user","content":[{"type":"text",
            "text":"Perform a web search for the query: <query>"}]}],
        "tools": [{"type":"web_search_20250305","name":"web_search","max_uses":5}]
      }

  它期望响应里含 `{"type":"web_search_tool_result"}` 内容块，其中的
  `web_search_result` 项提供 url/title/page_age，而 snippet 来自同一响应中
  `text` 块 `citations[].cited_text`（按 url 关联）。

  该插件默认打 api.deepseek.com 并要求 DEEPSEEK_API_KEY。本模块把同一个
  wire format 桥接到 WorkBuddy 的 `/agenttool/v1/search`，从而**复用本机
  已有的 WorkBuddy 登录态**，无需 DeepSeek API key、无需修改 DSH 客户端。

依赖：requests 或 httpx（二者皆可）；凭据解密见 workbuddy_atrest_crypto。
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:  # pragma: no cover - 加密链路可选
    from .workbuddy_atrest_crypto import decrypt_auth_field, is_encrypted_field
except ImportError:  # pragma: no cover - 允许脚本方式直接运行
    try:
        from workbuddy_atrest_crypto import decrypt_auth_field, is_encrypted_field
    except ImportError:
        decrypt_auth_field = None  # type: ignore[assignment]
        is_encrypted_field = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

WORKBUDDY_SEARCH_PATH = "/agenttool/v1/search"
WORKBUDDY_FETCH_PATH = "/agenttool/v1/webfetch"

DEFAULT_ENDPOINT = "https://www.workbuddy.ai"
DEFAULT_TIMEOUT = 20.0
DEFAULT_MAX_RESULTS = 8
DEFAULT_SEARCH_MODEL = "deepseek-v4-flash"

USER_AGENT = "codebuddy2api-search-bridge/1.0"

# WorkBuddy 的 freshness 语法：d1..d30 / m1..m12 / y1..y5
_FRESHNESS_RE = re.compile(r"^(?:d(?:[1-9]|[12]\d|30)?|m(?:[1-9]|1[0-2])?|y[1-5]?)$")


class SearchError(RuntimeError):
    """搜索桥接失败（供上层映射成 WEB_PROVIDER_ERROR 语义）。"""


# ---------------------------------------------------------------------------
# 凭据
# ---------------------------------------------------------------------------


def default_auth_paths() -> list[Path]:
    """WorkBuddy / CodeBuddy 登录态文件的常见位置（新→旧）。"""
    home = Path.home()
    local = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
    cands: list[Path] = []

    for root in (
        local / "CodeBuddyExtension" / "Data" / "Public" / "auth",
        home / ".codebuddy" / "auth",
        home / ".workbuddy-ai" / "auth",
    ):
        if root.is_dir():
            cands.extend(root.glob("*.info"))

    env = os.environ.get("WORKBUDDY_AUTH_FILE")
    if env:
        cands.insert(0, Path(env))

    # 按 mtime 倒序（与 codebuddy2api 的 fix/fa5002e 一致：不按文件名排序）
    uniq: dict[str, Path] = {}
    for p in cands:
        try:
            if p.is_file():
                uniq[str(p).lower()] = p
        except OSError:
            continue
    return sorted(uniq.values(), key=lambda p: p.stat().st_mtime, reverse=True)


def _decode(value: Any) -> str:
    """解密 $wbEncrypted 信封；明文则原样返回。"""
    if value is None:
        return ""
    if isinstance(value, dict):
        if decrypt_auth_field is None:
            raise SearchError(
                "auth 文件中的凭据是 $wbEncrypted 加密格式，但无法导入 "
                "workbuddy_atrest_crypto（需要 cryptography 依赖）"
            )
        try:
            return decrypt_auth_field(value)
        except Exception as e:  # noqa: BLE001
            raise SearchError(f"解密 WorkBuddy 凭据失败：{e}") from e
    return str(value)


@dataclass
class WorkBuddySession:
    """一份可用的 WorkBuddy 登录态。"""

    access_token: str
    refresh_token: str = ""
    domain: str = ""
    endpoint: str = DEFAULT_ENDPOINT
    uid: str = ""
    enterprise_id: str = ""
    expires_at: int = 0
    path: Path | None = None

    @property
    def expired(self) -> bool:
        return bool(self.expires_at) and time.time() * 1000 >= (self.expires_at - 60_000)

    def headers(self) -> dict[str, str]:
        h = {
            "Content-Type": "application/json;charset=UTF-8",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token}",
            "X-Requested-With": "XMLHttpRequest",
            "User-Agent": USER_AGENT,
        }
        if self.uid:
            h["X-User-Id"] = self.uid
        if self.enterprise_id:
            h["X-Enterprise-Id"] = self.enterprise_id
            h["X-Tenant-Id"] = self.enterprise_id
        if self.domain:
            h["X-Domain"] = self.domain
        return h


def load_session(path: Path | None = None) -> WorkBuddySession:
    """读取并解密 WorkBuddy 登录态。"""
    candidates = [path] if path else default_auth_paths()
    errors: list[str] = []
    for p in candidates:
        if p is None or not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            errors.append(f"{p}: {e}")
            continue

        auth = data.get("auth") or {}
        account = data.get("account") or {}
        token = _decode(auth.get("accessToken"))
        if not token:
            errors.append(f"{p}: 无 accessToken")
            continue

        domain = (auth.get("domain") or "").strip()
        endpoint = os.environ.get("WORKBUDDY_SEARCH_ENDPOINT") or (
            f"https://{domain}" if domain else DEFAULT_ENDPOINT
        )
        return WorkBuddySession(
            access_token=token,
            refresh_token=_decode(auth.get("refreshToken")),
            domain=domain,
            endpoint=endpoint.rstrip("/"),
            uid=str(account.get("uid") or ""),
            enterprise_id=str(account.get("enterpriseId") or ""),
            expires_at=int(auth.get("expiresAt") or 0),
            path=p,
        )

    detail = "；".join(errors[:3]) if errors else "未找到任何 auth 文件"
    raise SearchError(
        f"找不到可用的 WorkBuddy 登录态（{detail}）。"
        "请先登录 WorkBuddy 客户端，或用 WORKBUDDY_AUTH_FILE 指定 auth 文件。"
    )


# ---------------------------------------------------------------------------
# 上游调用
# ---------------------------------------------------------------------------


def _post_json(url: str, headers: dict[str, str], body: dict, timeout: float) -> dict:
    """POST JSON 并解析响应；把 HTTP/网络错误转成 SearchError。"""
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            status = resp.status
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:400]
        except Exception:  # noqa: BLE001
            pass
        if e.code == 401:
            raise SearchError(
                f"WorkBuddy 搜索鉴权失败（HTTP 401）：登录态可能已失效，"
                f"请重新登录 WorkBuddy 客户端。{detail}"
            ) from e
        raise SearchError(f"WorkBuddy 搜索返回 HTTP {e.code}：{detail}") from e
    except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
        raise SearchError(f"无法连接 WorkBuddy 搜索端点 {url}：{e}") from e

    if status != 200:
        raise SearchError(f"WorkBuddy 搜索返回 HTTP {status}")

    try:
        payload = json.loads(raw)
    except ValueError as e:
        raise SearchError(f"WorkBuddy 搜索返回了非 JSON 响应：{raw[:200]}") from e

    # WorkBuddy 用 code 表示业务错误（0/缺失为成功）
    code = payload.get("code")
    if code not in (None, 0):
        raise SearchError(
            f"WorkBuddy 搜索业务错误 code={code}：{payload.get('msg') or '未知错误'}"
        )
    return payload


@dataclass
class SearchOutcome:
    """一次搜索的归一化结果。"""

    query: str
    results: list[dict] = field(default_factory=list)
    provider: Any = None
    total_results: int = 0
    elapsed_ms: int = 0

    def sources(self) -> list[dict]:
        out = []
        for r in self.results:
            item: dict[str, Any] = {"url": r.get("url", "")}
            if r.get("title"):
                item["title"] = r["title"]
            if r.get("snippet"):
                item["snippet"] = r["snippet"]
            if r.get("publishedAt"):
                item["publishedAt"] = r["publishedAt"]
            out.append(item)
        return out


def search(
    query: str,
    *,
    session: WorkBuddySession | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    freshness: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> SearchOutcome:
    """调用 WorkBuddy `/agenttool/v1/search`。

    blocked_domains 按 WorkBuddy 客户端自身的行为转成查询串内的 `-site:` 负向词
    （该端点没有 blocked_domains 字段）。
    """
    if not query or not query.strip():
        raise SearchError("搜索查询不能为空")

    sess = session or load_session()

    q = query.strip()
    if blocked_domains:
        suffix = " ".join(f"-site:{d}" for d in blocked_domains if d)
        if suffix:
            q = f"{q} ({suffix})"

    body: dict[str, Any] = {
        "query": q,
        "type": "text2text",
        "max_results": max(1, min(int(max_results), 50)),
    }
    if allowed_domains:
        body["allowed_domains"] = list(allowed_domains)
    if freshness:
        if not _FRESHNESS_RE.match(freshness):
            raise SearchError(
                f"freshness 取值非法：{freshness!r}（应为 d1..d30 / m1..m12 / y1..y5）"
            )
        body["freshness"] = freshness

    started = time.time()
    payload = _post_json(
        f"{sess.endpoint}{WORKBUDDY_SEARCH_PATH}", sess.headers(), body, timeout
    )

    results: list[dict] = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        url = (item.get("url") or "").strip()
        if not url:
            continue
        results.append(
            {
                "url": url,
                "title": item.get("title") or "",
                "snippet": item.get("snippet") or "",
            }
        )

    return SearchOutcome(
        query=query,
        results=results,
        provider=payload.get("provider"),
        total_results=int(payload.get("total_results") or len(results)),
        elapsed_ms=int(payload.get("response_time_ms") or (time.time() - started) * 1000),
    )


def fetch(url: str, *, session: WorkBuddySession | None = None, timeout: float = 30.0) -> dict:
    """调用 WorkBuddy `/agenttool/v1/webfetch`（保留供 DSH web_fetch 复用）。"""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise SearchError(f"仅支持 http(s) URL：{url!r}")
    sess = session or load_session()
    return _post_json(
        f"{sess.endpoint}{WORKBUDDY_FETCH_PATH}",
        sess.headers(),
        {"url": url},
        timeout,
    )


# ---------------------------------------------------------------------------
# Anthropic Messages ⇄ WorkBuddy 适配（供本地搜索网关使用）
# ---------------------------------------------------------------------------


def build_search_query(messages: list[dict]) -> str:
    """从 Anthropic Messages 请求中抽取真实搜索词。

    插件固定发送 `Perform a web search for the query: <query>`，这里剥掉该前缀，
    以免把提示语本身当成搜索词传给 WorkBuddy。
    """
    chunks: list[str] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    chunks.append(str(block.get("text") or ""))
    text = "\n".join(c for c in chunks if c).strip()

    prefix = "Perform a web search for the query:"
    if text.startswith(prefix):
        text = text[len(prefix):].strip()
    return text


def to_anthropic_messages_response(
    outcome: SearchOutcome, *, model: str = DEFAULT_SEARCH_MODEL
) -> dict:
    """把搜索结果包成 DSH 插件期望的 Anthropic Messages 响应。

    关键：snippet 必须放在 `text` 块的 `citations[].cited_text` 里，
    因为 mapAnthropicResponse 只从那里取 snippet（web_search_result 本身没有该字段）。
    """
    tool_results = [
        {
            "type": "web_search_result",
            "url": r["url"],
            "title": r.get("title") or r["url"],
            "page_age": r.get("publishedAt") or None,
        }
        for r in outcome.results
    ]

    citations = [
        {"type": "web_search_result_location", "url": r["url"], "cited_text": r["snippet"]}
        for r in outcome.results
        if r.get("snippet")
    ]

    content: list[dict] = [
        {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_bridge", "content": tool_results}
    ]
    if citations:
        content.append({"type": "text", "text": "", "citations": citations})

    return {
        "id": "msg_bridge_" + os.urandom(8).hex(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
