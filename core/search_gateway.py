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

用法（正常路径——搜索已并入代理端口，无需单独起进程）：
    python -m core.converter --port 8787
    # 搜索端点即 POST http://127.0.0.1:8787/v1/searchGateway/messages

然后在 ~/.dsh/profiles/desktop/cordis.patch.yml 中：
    - id: web-search-deepseek
      name: "@deepseek-ai/dsh-web-search-deepseek"
      config:
        apiKey: local-bridge          # 占位值，仅用于让 available() 通过
        baseURL: http://127.0.0.1:8787/v1/searchGateway
        # 插件会拼成 .../v1/searchGateway/messages（后缀 /messages 是插件硬编码的）

独立调试模式（可选，单独监听一个端口）：
    python -m core.search_gateway --port 8790
    # 此时端点即 POST http://127.0.0.1:8790/messages
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

# FastAPI 相关必须在模块顶层导入：见 make_search_router 的注释
# （本模块启用了 `from __future__ import annotations`，注解按模块全局解析）。
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

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


class SearchRequestError(Exception):
    """请求本身不合法（缺工具声明、无查询词、JSON 坏）→ HTTP 400。"""

    def __init__(self, message: str, status: int = 400, err_type: str = "invalid_request_error"):
        super().__init__(message)
        self.message = message
        self.status = status
        self.err_type = err_type


def has_web_search_tool(body: dict) -> bool:
    """请求里是否声明了 web_search（托管型或 function 型皆认）。"""
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        ttype = str(tool.get("type") or "")
        if ttype.startswith("web_search") or tool.get("name") == "web_search":
            return True
    return False


def handle_search_request(
    body: dict, *, max_results: int = DEFAULT_MAX_RESULTS
) -> dict:
    """搜索桥的**传输无关**核心：Anthropic Messages 请求体 → 响应体。

    不依赖 HTTP 框架，供 FastAPI 路由与（测试用的）独立 HTTP server 共用。
    失败时抛 SearchRequestError（携带应回的 HTTP 状态码）。
    """
    if not isinstance(body, dict):
        raise SearchRequestError("请求体必须是 JSON 对象")

    if not has_web_search_tool(body):
        # 没有 web_search 工具：调用方没走搜索语义（可能误把这个路径当对话端点）。
        raise SearchRequestError(
            "请求缺少 web_search 工具声明；该端点只服务 DSH 的 web_search，"
            "不转发普通对话（对话请用 /v1/messages 或 /v1/responses）。"
        )

    query = build_search_query(body.get("messages") or [])
    if not query:
        raise SearchRequestError("无法从请求中解析出搜索词")

    started = time.time()
    try:
        outcome = search(query, max_results=max_results)
    except SearchError as e:
        _log(f"搜索失败 query={query!r}: {e}")
        # 401 类鉴权问题按 authentication_error 返回，便于排查
        status = 401 if "401" in str(e) else 502
        raise SearchRequestError(
            str(e), status, "authentication_error" if status == 401 else "api_error"
        ) from e
    except Exception as e:  # noqa: BLE001
        _log(f"搜索异常 query={query!r}: {e!r}")
        raise SearchRequestError(f"搜索桥接内部错误：{e}", 502, "api_error") from e

    model = str(body.get("model") or "deepseek-v4-flash")
    payload = to_anthropic_messages_response(outcome, model=model)
    _log(
        f"OK query={query!r} results={len(outcome.results)} "
        f"upstream_ms={outcome.elapsed_ms} total_ms={int((time.time() - started) * 1000)}"
    )
    return payload


def health_payload() -> tuple[int, dict]:
    """健康检查的（状态码, 响应体）。"""
    try:
        sess = load_session()
        return 200, {
            "status": "ok",
            "endpoint": sess.endpoint,
            "auth_file": str(sess.path) if sess.path else None,
            "token_expired": sess.expired,
        }
    except SearchError as e:
        return 503, {"status": "error", "message": str(e)}


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
        return has_web_search_tool(body)

    # ---- 路由 ----

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path in ("/health", "/healthz"):
            status, payload = health_payload()
            self._send(status, payload)
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

        try:
            payload = handle_search_request(
                body,
                max_results=int(
                    self.server.max_results  # type: ignore[attr-defined]
                    or DEFAULT_MAX_RESULTS
                ),
            )
        except SearchRequestError as e:
            self._error(e.status, e.message, e.err_type)
            return

        self._send(200, payload)


# ---------------------------------------------------------------------------
# FastAPI 挂载（把搜索并入主代理端口，不再单独监听）
# ---------------------------------------------------------------------------

# 默认挂载路径。必须以 /messages 结尾——插件的 endpoint 是
# `${baseURL}/messages` 硬编码的，所以 baseURL 要指到这一层的上一层。
SEARCH_ROUTE_PREFIX = "/v1/searchGateway"


def make_search_router(*, max_results: int = DEFAULT_MAX_RESULTS):
    """构造搜索能力的 FastAPI APIRouter，供 converter 挂到主 app 上。

    路径：POST {prefix}/messages 与 GET {prefix}/health
    挂上后 cordis.patch.yml 里 baseURL 写 `http://127.0.0.1:8787/v1/searchGateway`。
    """
    # 注意：本模块有 `from __future__ import annotations`，函数注解会变成字符串，
    # FastAPI 需要按**模块全局**解析它们。所以 Request 必须在模块顶层导入
    # （放在这里当局部名会解析失败，被误当成 query 参数 → 422）。
    router = APIRouter()

    @router.get(SEARCH_ROUTE_PREFIX + "/health")
    def search_health():
        status, payload = health_payload()
        return JSONResponse(status_code=status, content=payload)

    @router.post(SEARCH_ROUTE_PREFIX + "/messages")
    async def search_messages(request: Request):
        try:
            body = await request.json()
        except Exception as e:  # noqa: BLE001 - 非法 JSON
            return JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": f"请求体不是合法 JSON：{e}"},
                },
            )
        try:
            payload = handle_search_request(body, max_results=max_results)
        except SearchRequestError as e:
            return JSONResponse(
                status_code=e.status,
                content={"type": "error", "error": {"type": e.err_type, "message": e.message}},
            )
        return JSONResponse(status_code=200, content=payload)

    return router


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
    _log(
        "接入方式       : cordis.patch.yml → web-search-deepseek.baseURL "
        f"指向 http://{args.host}:{args.port}"
    )
    _log("注意           : 该独立模式仅供调试；正常使用请直接用 converter（搜索已并入其端口）")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        _log("收到中断，退出")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
