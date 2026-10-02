"""Remove client identity declarations from system/developer text only."""

import re

# 后端（copilot / workbuddy）要求 messages[0] 必须是 system prompt，
# 否则直接返回 11128 "first message is not system prompt"。
# 因此：过滤后若 system 内容为空，用中性占位替换，而不是整条丢弃。
_SYSTEM_FALLBACK = "You are a helpful assistant."


_DECLARATION = re.compile(
    r"^(?:you\s+are\b|your\s+(?:designated\s+)?identity\b|"
    r"you\s+have\s+been\s+invoked\b|you\s+are\s+powered\s+by\b|"
    r"你(?:是|的身份是|的角色是|由)|您(?:是|的身份是|的角色是|由))",
    re.IGNORECASE,
)
_IDENTITY = re.compile(
    r"\b(?:zcode|codex|claude(?:\s+code)?|codebuddy|workbuddy|"
    r"opencode|sisyphus|anthropic|openai|chatgpt|gemini|"
    r"deepseek|qwen|kimi|glm|gpt|AI|LLM|assistant|agent|model)\b|"
    r"人工智能|智能助手|编程助手|编码助手|语言模型|智能体",
    re.IGNORECASE,
)
_OBLIGATION = re.compile(
    r"^you\s+are\s+(?:required|expected|allowed|not allowed|forbidden|responsible|"
    r"operating|working|running|using|to)\b", re.IGNORECASE
)
_HEADINGS = re.compile(
    r"^#{1,6}\s+(?:ZCode|Codex|Claude Code|CodeBuddy|WorkBuddy|OpenCode)\s+",
    re.IGNORECASE,
)
_BRANCH_TEMPLATE = re.compile(
    r"Main branch \(you will usually use this for PRs\):", re.IGNORECASE
)


def filter_system_text(text: str) -> str:
    lines = []
    for line in text.splitlines(keepends=True):
        candidate = re.sub(r"^\s*(?:[-*]\s+|#{1,6}\s+)?", "", line)
        # Limit removal to declarative sentences. Keep subsequent instructions.
        pieces = re.split(r"(?<=[.!?。！？])(?=\s|$)", candidate)
        kept = []
        removed = False
        for piece in pieces:
            clean = piece.strip()
            if _DECLARATION.search(clean) and _IDENTITY.search(clean) and not _OBLIGATION.search(clean):
                removed = True
            else:
                kept.append(piece)
        if removed:
            line = "".join(kept).lstrip()
            if not line.strip():
                continue
        # Preserve branch value and section meaning, dropping fixed client wording.
        line = _BRANCH_TEMPLATE.sub("Main branch:", line)
        line = _HEADINGS.sub("# ", line)
        lines.append(line)
    return "".join(lines)


def filter_system_identity(body: dict, fallback: str = _SYSTEM_FALLBACK) -> dict:
    """Return a new body; never modify user/assistant/tool messages or tool schemas.

    fallback: 首条不是 system 时补的占位内容（默认中性提示词，
    可由调用方替换成运维方配置的自定义提示词）。
    """
    messages = []
    for message in body.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") not in ("system", "developer"):
            messages.append(message)
            continue
        content = message.get("content")
        if isinstance(content, str):
            filtered = filter_system_text(content)
            messages.append(
                dict(message, content=filtered if filtered.strip() else fallback)
            )
        elif isinstance(content, list):
            blocks = []
            for block in content:
                if isinstance(block, dict) and block.get("type") in ("text", "input_text", "output_text") and isinstance(block.get("text"), str):
                    filtered = filter_system_text(block["text"])
                    if filtered.strip():
                        blocks.append(dict(block, text=filtered))
                else:
                    blocks.append(block)
            messages.append(
                dict(message, content=blocks if blocks else fallback)
            )
        else:
            messages.append(message)
    # 兜底：确保首条为 system（否则后端 11128 拒绝）
    if not messages or not isinstance(messages[0], dict) or messages[0].get("role") not in ("system", "developer"):
        messages.insert(0, {"role": "system", "content": fallback})
    return dict(body, messages=messages)


SYSTEM_PROMPT_MODES = ("fallback", "prepend", "replace")


def apply_system_prompt(body: dict, prompt: str, mode: str = "fallback") -> dict:
    """按 mode 应用自定义的首条系统提示词（仅在 prompt 非空时调用）。

    - fallback：仅当客户端没发 system、或 system 被过滤成空时才用 prompt
    - prepend ：把 prompt 插到最前，保留客户端原有的 system
    - replace ：丢弃客户端所有 system，只发 prompt

    注意：prompt 是运维方显式配置的内容，**不参与身份过滤**。否则像
    "You are a helpful assistant." 这种会被 filter_system_text 判为身份声明
    而过滤成空，配置就静默失效了。
    """
    if mode not in SYSTEM_PROMPT_MODES:
        raise ValueError(
            f"未知的 system_prompt_mode：{mode!r}（可选：{'/'.join(SYSTEM_PROMPT_MODES)}）"
        )
    if mode == "fallback":
        return filter_system_identity(body, fallback=prompt)

    out = filter_system_identity(body)
    msgs = list(out.get("messages") or [])
    if mode == "replace":
        msgs = [
            m
            for m in msgs
            if not (isinstance(m, dict) and m.get("role") in ("system", "developer"))
        ]
    msgs.insert(0, {"role": "system", "content": prompt})
    return dict(out, messages=msgs)
