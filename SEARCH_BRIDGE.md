# 用 WorkBuddy 的搜索能力增强 DSH 的 web_search

**目标**：让 DSH（DeepSeek Harness）的 `web_search` 工具使用本机 **WorkBuddy 登录态**去搜索，
不需要 DeepSeek API key，也不需要修改 DSH 客户端本体。

---

## 1. 为什么不能只靠 codebuddy2api 代理桥接

代理（`core/converter.py`）替换的是**模型**链路，而搜索走的是**另一条独立链路**：

| 能力 | 走的协议 | 是否经过代理 |
|---|---|---|
| 对话/工具调用 | OpenAI Responses / Chat → `workbuddy-proxy`(8787) | ✅ 是 |
| `web_fetch` | DSH 本地 HTTP（DNS 校验 + IP 绑定），无凭据 | ❌ 否 |
| **`web_search`** | **DSH 执行时另发一个 Anthropic Messages 请求** | ❌ 否 |

DSH 的 `web_search` 是**宿主侧本地工具**：模型只负责"调用它"，真正执行由
`@deepseek-ai/dsh-web-search-deepseek` 插件完成——它在执行时向
`{baseURL}/messages` 发一个**独立的** Anthropic Messages 请求：

```
POST {baseURL}/messages
{
  "model": "deepseek-v4-flash", "max_tokens": 4096,
  "messages": [{"role":"user","content":[{"type":"text",
      "text":"Perform a web search for the query: <用户查询>"}]}],
  "tools": [{"type":"web_search_20250305","name":"web_search","max_uses":5}]
}
```

所以**代理无法用 `tools` 字段把搜索带过去**（DSH 从不向模型发送 hosted 工具，
`@earendil-works/pi-ai` 只产出 `type:"function"`）。

**但好消息是**：这个链路的端点由 `baseURL` 配置，而且插件在
`apiKey` 非空时即认为 provider 可用。于是我们可以**在同一个 wire format 上做替换**：
写一个本地网关冒充"Anthropic Messages + web_search"，实际去调 WorkBuddy 的搜索接口。

> 这也解释了最初的故障：DSH 客户端零改动下，插件默认打 `api.deepseek.com` 且
> 要求 `DEEPSEEK_API_KEY`，因此报
> `DeepSeek search has no API key for "DEEPSEEK_API_KEY"`。

---

## 2. 架构

```
DSH 会话（零改动）
   └─ web_search 工具（dsh-tool-web）
        └─ dsh-web-search-deepseek 插件   ← 只改这个插件的 config
             │  POST {baseURL}/messages  （Anthropic Messages + web_search_20250305）
             │  = POST 127.0.0.1:8787/v1/searchGateway/messages
             ▼
      converter 主 app 上的搜索路由  core/search_gateway.py
             │  取出真实查询词、剥掉插件前缀
             ▼
      WorkBuddy  core/workbuddy_search.py
             │  POST https://www.workbuddy.ai/agenttool/v1/search
             │  Authorization: Bearer <本机 WorkBuddy 登录态>
             ▼
        真实搜索结果 ──► 包成 web_search_tool_result + text.citations ──► 返回 DSH
```

搜索与对话（`/v1/messages`、`/v1/chat/completions`、`/v1/responses`）共用
同一个 app 与同一个端口，靠**路径前缀**区分，互不干扰。

### 关键实现细节

- **snippet 必须放在 `citations[].cited_text`**：插件的 `mapAnthropicResponse()`
  只从 `text` 块的 citations 里取 snippet（`web_search_result` 本身没有该字段）。
  放错位置会导致搜索结果"有链接没摘要"。
- **必须剥掉 `Perform a web search for the query:` 前缀**，否则会把提示语当搜索词。
- **`apiKey` 只是占位值**：插件的 `available()` 形如
  `((apiKey?.length ?? 0) > 0 || resolveApiKey || resolveAccountToken) && URL.canParse(baseURL) && ...`
  ——只要 key 非空、baseURL 合法、maxTokens/maxUses 为正整数，provider 就可用。
  真正鉴权发生在网关侧（WorkBuddy Bearer token）。
- **凭据解密复用 codebuddy2api 的成果**：WorkBuddy 5.6+ 的 `.info` 里
  `accessToken` 是 `{"$wbEncrypted":1,...}` 信封，由
  `core/workbuddy_atrest_crypto.py` 解密（`loggerGet()` →
  `sha256(atRestSecretKey)` → AES-256-GCM）。本机实测 keyId `9127dea1b44020a7`
  与 `~/.workbuddy-ai/keyblob` 的 `protectorKeyId` 一致。

---

## 3. 部署步骤

### 3.1 启动（搜索已并入代理端口，只监听一个端口）

搜索网关是主 app 上的一个路由，**不需要单独起进程，也不额外占端口**：

```powershell
cd O:\SynologyDrive\GitHub\codebuddy2api
.\.venv\Scripts\python.exe -m core.converter --port 8787
# → 全部能力都在 8787：
#    POST /v1/messages                    对话（Anthropic）
#    POST /v1/chat/completions            对话（OpenAI）
#    POST /v1/responses                   对话（Responses）
#    POST /v1/searchGateway/messages      DSH web_search ← 新增
#    GET  /v1/searchGateway/health        搜索健康检查
```

启动输出会列出搜索端点：

```
✅ 监听 http://127.0.0.1:8787（直连后端，原生 function calling）
   ...
   搜索网关  : http://127.0.0.1:8787/v1/searchGateway（供 DSH web-search-deepseek.baseURL 使用）
```

**关闭**：`--no-search-gateway`（此时 DSH 的 `web_search` 退回官方 DeepSeek 端点）。

> **为什么路径必须以 `/messages` 结尾**：插件的端点是硬编码的
> `` `${baseURL}/messages` ``，配置改不了。所以 `baseURL` 要指到
> `/v1/searchGateway` 这一层，插件才会拼出 `/v1/searchGateway/messages`。

> **为什么不用 `/v1` 做前缀**：那样插件会打到 `/v1/messages`，而该路径已被
> Anthropic 对话端点占用（Claude Code / CC Switch 在用），两者协议形状不同
> （对话 vs `web_search_20250305`），共用会互相干扰。所以搜索挂在独立前缀下。

独立调试模式（可选，单独监听一个端口，仅供排障）：

```powershell
.\.venv\Scripts\python.exe -m core.search_gateway --port 8790
```

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8787/v1/searchGateway/health
# {"status":"ok","endpoint":"https://www.workbuddy.ai","auth_file":"...","token_expired":false}
```

> 搜索路由挂载失败（或未启用）都**不影响代理主功能**。

### 3.2 配置 DSH 插件

编辑 `~/.dsh/profiles/desktop/cordis.patch.yml`：

```yaml
- id: web-search-deepseek
  name: "@deepseek-ai/dsh-web-search-deepseek"
  config:
    apiKey: local-bridge          # 占位值，仅用于让插件的 available() 通过
    baseURL: http://127.0.0.1:8787/v1/searchGateway
    # 插件会拼成 .../v1/searchGateway/messages（后缀 /messages 是插件硬编码的）
```

改完重启 DSH（或在支持 HMR 的部分生效后重开）。

### 3.3 验证

在 DSH 里直接调用 `web_search`，或复刻插件报文验证：

```powershell
$body = @{
  model = "deepseek-v4-flash"; max_tokens = 4096
  messages = @(@{ role="user"; content=@(@{ type="text"
      text="Perform a web search for the query: test query" }) })
  tools = @(@{ type="web_search_20250305"; name="web_search"; max_uses=5 })
} | ConvertTo-Json -Depth 8
Invoke-RestMethod -Uri http://127.0.0.1:8787/v1/searchGateway/messages -Method POST `
  -Body $body -ContentType "application/json" | ConvertTo-Json -Depth 6
```

---

## 4. 本机实测结果

搜索日志（DSH 会话内直接调用 `web_search`，客户端零改动，单端口）：

```
[search-gateway] OK query='Redis persistence RDB vs AOF' results=8 upstream_ms=1858 total_ms=2258
[search-gateway] OK query='FastAPI dependency injection' results=8 upstream_ms=1858 total_ms=2258
```

同一端口上既有端点未受影响（实测均 200）：

```
POST /v1/messages            200   对话（Anthropic）
POST /v1/chat/completions    200   对话（OpenAI）
GET  /v1/models              200   模型列表
POST /v1/searchGateway/messages  200   搜索（新增）
GET  /v1/searchGateway/health    200   搜索健康检查
```

搜索可用性对照：

| 组件 | 改造前 | 改造后 |
|---|---|---|
| `web_search` | ❌ `no API key for "DEEPSEEK_API_KEY"`（日志中 253 次） | ✅ 8 条真实结果，~2s |
| `web_fetch` | ✅ 一直可用（无需凭据） | ✅ 不变 |

---

## 5. 可选参数

`search()` 支持（网关按需透传）：

| 参数 | 说明 |
|---|---|
| `max_results` | 结果条数，网关侧 `--max-results` 限制上限 |
| `allowed_domains` | 只在指定域名内搜（WorkBuddy 原生字段） |
| `blocked_domains` | WorkBuddy 端点**没有**该字段，网关转成查询串里的 `-site:x.com` 负向词 |
| `freshness` | `d1..d30` / `m1..m12` / `y1..y5`；非法值会被拒绝 |

---

## 6. 局限与注意事项

1. **依赖 WorkBuddy 登录态有效性**。token 过期需重新登录 WorkBuddy 客户端。
   本机实测 `expiresAt` 还有约 355 天，且 `.info` 与 `keyblob` 都在本机，
   没有额外的刷新链路（codebuddy2api 的 `CredentialManager._refresh()` 走的是
   `/v2/plugin/auth/token/refresh`，如需长期无人值守可复用）。
2. **`agentic_search` 未接入**。WorkBuddy 另有
   `POST /agenttool/v1/agentic_search`（SSE，`search_mode:2`），
   但官方说明"currently supports financial and investment topics only"，
   通用性不如 `/agenttool/v1/search`，故未采用。
3. **`web_fetch` 未改动**。它本来就能用；若要也走 WorkBuddy 的
   `/agenttool/v1/webfetch`（`{url, prompt}`），可基于 `fetch()` 再包一层。
4. **搜索路由随 app 一起暴露**。若用 `--host 0.0.0.0` 对外提供服务，
   `/v1/searchGateway/*` 也会一起对外开放且**无独立鉴权**（受主服务的
   `--api-key` 约束范围与其它端点一致）。仅本机使用时保持默认 `127.0.0.1` 即可。
5. **不要提交 `apiKey` 到公开仓库**——虽然这里只是占位值。

---

## 7. 测试

```powershell
cd O:\SynologyDrive\GitHub\codebuddy2api
.\.venv\Scripts\python.exe -m pytest tests/test_workbuddy_search.py -q
# 72 passed
```

覆盖：查询词前缀剥离、Anthropic 响应形状（含"插件能否还原 snippet"的仿射验证）、
参数映射、freshness 校验、凭据明文/加密两条路径、
**FastAPI 挂载路径的端到端可达性**（用 TestClient 走完整 ASGI 栈，
能抓到"`Request` 注解解析失败被当成 query 参数 → 422"这类只在请求期暴露的问题）、
公共核心 `handle_search_request` 的错误码映射（400/401/502）、
独立调试模式的 HTTP 层（200/400/401/404/502、keep-alive 不串包、
`allow_reuse_address` 关闭、端口冲突明确报错），
以及**默认挂载**与 `--no-search-gateway` 关闭开关（用真实 argparse 解析验证）。
