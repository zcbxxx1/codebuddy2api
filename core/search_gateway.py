#!/usr/bin/env python3
"""
search_gateway — 把 DSH 的 web_search 桥接到本机 WorkBuddy 登录态的本地网关。

为什么要这个网关：
  DSH 的 `web_search` 是宿主侧工具，由 `@deepseek-ai/dsh-web-search-deepseek`
  插件在**执行时**请求 `{baseURL}/messages`（Anthropic Messages 协议 +
  `web_search_20250305` 服务端工具）。默认端点打 api.deepseek.com 且要求
  DEEPSEEK_API_KEY。

  本网关实现同一个 wire format，但把实际的搜索落到 WorkBuddy 的
  `/agenttool/v1/search`（复用本机已登录的 WorkBuddy 凭据）。
  这样 DSH **客户端零改动**：只需在 cordis.patch.yml 里给
  `web-search-deepseek` 配置 baseURL 指向本网关，并给一个占位 apiKey
  让插件的 available() 判定为真。

用法：
    python -m core.search_gateway --port 8790

然后在 ~/.dsh/profiles/desktop/cordis.patch.yml 中：
    - id: web-search-deepseek
      name: "@deepseek-ai/dsh-web-search-deepseek"
      config:
        apiKey: local-bridge          # 占位值，仅用于让 available() 通过
        baseURL: http://127.0.0.1:8790
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

try:
    from .workbuddy_search import (
        DEFAULT_MAX_RESULTS,
        SearchError,
        build_search_query,
        load_session,
        search,
        to_anthropic_messages_response,
    )
except ImportError:  # pragma: no cover - 支持脚本直跑
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.workbuddy_search import (  # type: ignore[no-redef]
        DEFAULT_MAX_RESULTS,
        SearchError,
        build_search_query,
        load_session,
        search,
        to_anthropic_messages_response,
    )


def _setup_console_encoding() -> None:
    """把 stdout/stderr 切成 UTF-8，避免 Windows GBK 控制台下中文日志乱码。

    Python 的 TextIOWrapper 是**惰性编码**的：编码错误要到 flush/write 落盘时才暴露，
    所以不能靠 try/except 在写日志时兜住——必须在启动时就把编码定好。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            enc = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
            if enc in ("utf8", "utf8mb4"):
                continue
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass


def _log(msg: str) -> None:
    try:
        sys.stderr.write(f"[search-gateway] {msg}\n")
        sys.stderr.flush()
    except (UnicodeEncodeError, ValueError):
        # 极端兜底：连 reconfigure 都不可用时，用转义形式保住信息
        try:
            enc = getattr(sys.stderr, "encoding", None) or "ascii"
            sys.stderr.write(
                f"[search-gateway] {msg.encode(enc, 'backslashreplace').decode(enc, 'replace')}\n"
            )
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass


class GatewayHandler(BaseHTTPRequestHandler):
    server_version = "codebuddy2api-search-gateway/1.0"
    protocol_version = "HTTP/1.1"

    # ---- 基础工具 ----

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str, err_type: str = "invalid_request_error") -> None:
        # 早退分支若还没读请求体，keep-alive 下会串包，这里兜底清空
        self._drain_body()
        self._send(status, {"type": "error", "error": {"type": err_type, "message": message}})

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        if self.server.verbose:  # type: ignore[attr-defined]
            _log(fmt % args)

    def _read_json(self) -> dict:
        self._body_drained = True
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError as e:
            raise SearchError(f"请求体不是合法 JSON：{e}") from e
        if not isinstance(data, dict):
            raise SearchError("请求体必须是 JSON 对象")
        return data

    def _drain_body(self) -> None:
        """读完并丢弃尚未消费的请求体，保证 keep-alive 连接不串包。幂等。"""
        if getattr(self, "_body_drained", False):
            return
        self._body_drained = True
        length = int(self.headers.get("Content-Length") or 0)
        if length > 0:
            try:
                self.rfile.read(length)
            except OSError:
                pass

    def _has_web_search_tool(self, body: dict) -> bool:
        for tool in body.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            ttype = str(tool.get("type") or "")
            if ttype.startswith("web_search") or tool.get("name") == "web_search":
                return True
        return False

    # ---- 路由 ----

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path in ("/health", "/healthz"):
            try:
                sess = load_session()
                self._send(
                    200,
                    {
                        "status": "ok",
                        "endpoint": sess.endpoint,
                        "auth_file": str(sess.path) if sess.path else None,
                        "token_expired": sess.expired,
                    },
                )
            except SearchError as e:
                self._send(503, {"status": "error", "message": str(e)})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path != "/messages":
            # 必须先把请求体读掉，否则 keep-alive 连接会串包
            self._drain_body()
            self._error(404, f"未知路径 {path}；本网关只实现 POST /messages")
            return

        try:
            body = self._read_json()
        except SearchError as e:
            self._error(400, str(e))
            return

        if not self._has_web_search_tool(body):
            # 没有 web_search 工具：说明调用方没走搜索语义（例如被误当聊天端点）。
            self._error(
                400,
                "请求缺少 web_search 工具声明；本网关只服务 DSH 的 web_search 能力，"
                "不转发普通对话（对话请用 /v1/responses 或 /v1/chat/completions）。",
            )
            return

        query = build_search_query(body.get("messages") or [])
        if not query:
            self._error(400, "无法从请求中解析出搜索词")
            return

        max_results = int(
            self.server.max_results  # type: ignore[attr-defined]
            or DEFAULT_MAX_RESULTS
        )

        started = time.time()
        try:
            outcome = search(query, max_results=max_results)
        except SearchError as e:
            _log(f"搜索失败 query={query!r}: {e}")
            # 401 类鉴权问题按 authentication_error 返回，便于排查
            status = 401 if "401" in str(e) else 502
            self._error(status, str(e), "authentication_error" if status == 401 else "api_error")
            return
        except Exception as e:  # noqa: BLE001
            _log(f"搜索异常 query={query!r}: {e!r}")
            self._error(502, f"搜索桥接内部错误：{e}", "api_error")
            return

        model = str(body.get("model") or "deepseek-v4-flash")
        payload = to_anthropic_messages_response(outcome, model=model)
        _log(
            f"OK query={query!r} results={len(outcome.results)} "
            f"upstream_ms={outcome.elapsed_ms} total_ms={int((time.time()-started)*1000)}"
        )
        self._send(200, payload)


class GatewayServer(ThreadingHTTPServer):
    """禁用地址复用：端口被占用时必须失败，而不是静默共存。

    http.server.HTTPServer 默认 allow_reuse_address = 1，在 Windows 上会让
    bind() 在端口已被监听时仍然成功——于是第二个实例能"看起来起来了"，
    实际请求仍被第一个进程接走（可能是个陈旧的旧版本）。
    搜索网关现在随代理默认启动，这种静默遮蔽必须避免，所以显式关掉。
    """

    allow_reuse_address = False
    daemon_threads = True

    max_results: int = DEFAULT_MAX_RESULTS
    verbose: bool = False


def build_server(
    host: str = "127.0.0.1",
    port: int = 8790,
    *,
    max_results: int = DEFAULT_MAX_RESULTS,
    verbose: bool = False,
) -> GatewayServer:
    """构造（但不启动）搜索网关 HTTP 服务。

    先做登录态预检——没有可用凭据就直接抛错，而不是等 DSH 侧报错。
    端口被占用时抛 OSError（见 GatewayServer 的 allow_reuse_address 说明）。
    供 `python -m core.search_gateway` 与 converter 的同进程内嵌两处复用。
    """
    load_session()  # 预检：无凭据即抛 SearchError
    httpd = GatewayServer((host, port), GatewayHandler)
    httpd.max_results = max_results
    httpd.verbose = verbose
    return httpd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="search_gateway",
        description="把 DSH 的 web_search 桥接到本机 WorkBuddy 登录态（Anthropic Messages 兼容）。",
    )
    ap.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    ap.add_argument("--port", type=int, default=8790, help="监听端口（默认 8790）")
    ap.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_MAX_RESULTS,
        help=f"每次搜索返回的最大结果数（默认 {DEFAULT_MAX_RESULTS}）",
    )
    ap.add_argument("--verbose", action="store_true", help="打印每个请求的访问日志")
    args = ap.parse_args(argv)

    _setup_console_encoding()

    # 启动前预检：没有可用登录态就直接失败，而不是等 DSH 报错
    try:
        sess = load_session()
    except SearchError as e:
        sys.stderr.write(f"启动预检失败：{e}\n")
        return 2

    httpd = build_server(
        args.host, args.port, max_results=args.max_results, verbose=args.verbose
    )

    _log(f"监听 http://{args.host}:{args.port}")
    _log(f"WorkBuddy 端点 : {sess.endpoint}")
    _log(f"登录态文件     : {sess.path}")
    _log(f"token 过期     : {sess.expired}")
    _log("接入方式       : cordis.patch.yml → web-search-deepseek.baseURL 指向本地址")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        _log("收到中断，退出")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
