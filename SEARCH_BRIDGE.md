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
             ▼
      本地搜索网关  core/search_gateway.py   (127.0.0.1:8790)
             │  取出真实查询词、剥掉插件前缀
             ▼
      WorkBuddy  core/workbuddy_search.py
             │  POST https://www.workbuddy.ai/agenttool/v1/search
             │  Authorization: Bearer <本机 WorkBuddy 登录态>
             ▼
        真实搜索结果 ──► 包成 web_search_tool_result + text.citations ──► 返回 DSH
```

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

### 3.1 启动（搜索网关默认随代理一起起）

网关已内嵌进 converter，**一个进程同时提供代理和搜索，且默认开启**：

```powershell
cd O:\SynologyDrive\GitHub\codebuddy2api
.\.venv\Scripts\python.exe -m core.converter --port 8787
# → 代理   http://127.0.0.1:8787
# → 搜索网关 http://127.0.0.1:8790   （默认端口，无需额外参数）
```

启动输出会明确列出两者：

```
✅ 监听 http://127.0.0.1:8787（直连后端，原生 function calling）
   ...
   搜索网关  : http://127.0.0.1:8790（供 DSH web-search-deepseek.baseURL 使用）
```

**关闭方式**（任一）：

| 方式 | 写法 |
|---|---|
| 命令行开关 | `--no-search-gateway` |
| 端口设 0 | `--search-gateway-port 0` |
| 环境变量 | `CODEBUDDY_SEARCH_GATEWAY_PORT=0` |

**换端口**：`--search-gateway-port 9100`（记得同步改 `cordis.patch.yml` 的 `baseURL`）。

绑定地址会自动收敛到回环：若 `--host 0.0.0.0`，网关仍只绑 `127.0.0.1`
（DSH 插件就在本机，无需对外暴露）。

也可以**单独**起网关进程（只想要搜索、不跑代理）：

```powershell
.\.venv\Scripts\python.exe -m core.search_gateway --port 8790
```

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8790/health
# {"status":"ok","endpoint":"https://www.workbuddy.ai","auth_file":"...","token_expired":false}
```

> **端口占用不会静默遮蔽**。`http.server.HTTPServer` 默认 `allow_reuse_address=1`，
> 在 Windows 上会让 `bind()` 到已占用端口仍然成功——于是第二个实例"看起来起来了"，
> 而请求其实被第一个（可能是旧版本的）进程接走。本网关的 `GatewayServer` 显式
> 关掉了地址复用：端口被占用时**明确报告**并跳过，不会假装成功。
>
> 网关因任何原因起不来（端口占用、无 WorkBuddy 凭据）都**不影响代理主功能**。

### 3.2 配置 DSH 插件

编辑 `~/.dsh/profiles/desktop/cordis.patch.yml`：

```yaml
- id: web-search-deepseek
  name: "@deepseek-ai/dsh-web-search-deepseek"
  config:
    apiKey: local-bridge          # 占位值，仅用于让插件的 available() 通过
    baseURL: http://127.0.0.1:8790
```

改完重启 DSH（或在支持 HMR 的部分生效后重开）。

### 3.3 验证

在 DSH 里直接调用 `web_search`，或复刻插件报文验证网关：

```powershell
$body = @{
  model = "deepseek-v4-flash"; max_tokens = 4096
  messages = @(@{ role="user"; content=@(@{ type="text"
      text="Perform a web search for the query: test query" }) })
  tools = @(@{ type="web_search_20250305"; name="web_search"; max_uses=5 })
} | ConvertTo-Json -Depth 8
Invoke-RestMethod -Uri http://127.0.0.1:8790/messages -Method POST -Body $body `
  -ContentType "application/json" | ConvertTo-Json -Depth 6
```

---

## 4. 本机实测结果

网关日志（DSH 会话内直接调用 `web_search`，客户端零改动）：

```
[search-gateway] 监听 http://127.0.0.1:8790
[search-gateway] WorkBuddy 端点 : https://www.workbuddy.ai
[search-gateway] 登录态文件     : C:\Users\13932\AppData\Local\CodeBuddyExtension\Data\Public\auth\workbuddy-desktop-ai.info
[search-gateway] OK query='Python asyncio best practices 2026' results=8 upstream_ms=7374
[search-gateway] OK query='Apache Kafka vs RabbitMQ 2026 comparison' results=8 upstream_ms=1938
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
4. **网关无鉴权**。仅监听 `127.0.0.1`，不要暴露到公网。
5. **不要提交 `apiKey` 到公开仓库**——虽然这里只是占位值。

---

## 7. 测试

```powershell
cd O:\SynologyDrive\GitHub\codebuddy2api
.\.venv\Scripts\python.exe -m pytest tests/test_workbuddy_search.py -q
# 66 passed
```

覆盖：查询词前缀剥离、Anthropic 响应形状（含"插件能否还原 snippet"的仿射验证）、
参数映射、freshness 校验、凭据明文/加密两条路径、网关 HTTP 层
（200/400/401/404/502、keep-alive 不串包）、`build_server` 工厂，
converter 内嵌启动（回环绑定、端口占用明确报错、意外不拖垮代理），
以及**默认开启**与三个关闭开关（用真实 argparse 解析验证）。
