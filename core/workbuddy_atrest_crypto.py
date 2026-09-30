#!/usr/bin/env python3
"""
workbuddy_atrest_crypto — 支持 WorkBuddy 5.6.0+ 的 $wbEncrypted 加密登录态（issue #23）。

背景：
  WorkBuddy 5.6.0 起，auth .info 文件中的 accessToken / refreshToken 等字段由明文字符串
  变为加密信封对象：
      {"$wbEncrypted": 1, "envelope": "<base64 of JSON>"}
  envelope JSON 结构（suite 1 / AES-256-GCM / sym-v1）：
      {"suite":1, "keyId":"<16 hex>", "nonce":"<b64 12B>",
       "authTag":"<b64 16B>", "ciphertext":"<b64>"}

密钥链路（已在本机 macOS WorkBuddy 5.6.x 实测验证）：
  1. protector key = sha256( atRestSecretKey字符串, utf8 )，32 字节。
     atRestSecretKey（44 字符 canonical base64）由 WorkBuddy Electron 原生层提供：
         process._linkedBinding("electron_browser_workbuddy_storage").loggerGet()
     返回 JSON：{"version":1, "atRestSecretKey":"...", "atRestDeveloperPublicKey":"..."}
     可通过 ELECTRON_RUN_AS_NODE=1 <WorkBuddy 可执行文件> 离线调用获取。
  2. keyId = sha256(protector_key).hex[:16]（即 envelope 中的 keyId，全安装静态一致）。
  3. field 场景 AAD（sym-v1, framing=field）：
         b"WB-AAD\\x00" + b"\\x01"
         + uint32be(5)  + b"WBEV1"          # STANDARD_FORMAT_ID.field
         + uint32be(6)  + b"sym-v1"         # scheme
         + uint32be(1)                      # suite
         + uint32be(16) + keyId(ascii)
         + b"\\x02"                          # FRAMING_CODE.field = 2
         + b"\\x00"                          # optionalUint64(sequence = undefined)
         + b"\\x00"                          # final = undefined -> 0
  4. AES-256-GCM，nonce 12B，authTag 16B，密文 = 明文（UTF-8）。

依赖：cryptography（pip install cryptography）
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError:  # pragma: no cover
    AESGCM = None

# ---------------------------------------------------------------------------
# 定位 WorkBuddy 可执行文件（用于 run-as-node 调用原生 loggerGet）
# ---------------------------------------------------------------------------

def _dedup(paths) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        try:
            key = str(p).lower()
        except Exception:  # noqa: BLE001
            continue
        if key in seen:
            continue
        seen.add(key)
        if p.is_file():
            out.append(p)
    return out


def _windows_candidates() -> list[Path]:
    """Windows 官方安装器（WorkBuddyAI）与旧命名（WorkBuddy）的常见位置。"""
    home = Path.home()
    local = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
    progfiles = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
    ]
    cands: list[Path] = []
    for base in progfiles:
        cands.append(base / "WorkBuddyAI" / "WorkBuddyAI.exe")
        cands.append(base / "WorkBuddy" / "WorkBuddy.exe")
    cands.append(local / "Programs" / "WorkBuddyAI" / "WorkBuddyAI.exe")
    cands.append(local / "Programs" / "WorkBuddy" / "WorkBuddy.exe")
    found = _dedup(cands)
    if found:
        return found
    # 兜底：在常见安装根目录下按名字扫描一层子目录
    scanned: list[Path] = []
    for base in [local / "Programs", *progfiles]:
        if not base.is_dir():
            continue
        try:
            for sub in base.iterdir():
                if sub.is_dir() and sub.name.lower().startswith("workbuddy"):
                    scanned.extend(sub.glob("WorkBuddy*.exe"))
        except OSError:
            continue
    return _dedup(scanned)


def _electron_candidates() -> list[Path]:
    env = os.environ.get("WORKBUDDY_ELECTRON_PATH")
    cands: list[Path] = [Path(env)] if env else []
    home = Path.home()
    if sys.platform == "darwin":
        cands.append(Path("/Applications/WorkBuddy.app/Contents/MacOS/Electron"))
        cands.append(Path("/Applications/WorkBuddyAI.app/Contents/MacOS/WorkBuddyAI"))
    elif sys.platform == "win32":
        cands.extend(_windows_candidates())
    else:
        cands.append(Path("/opt/WorkBuddy/workbuddy"))
        cands.append(Path("/usr/bin/workbuddy"))
    return _dedup(cands)


_LOGGERGET_JS = (
    "const b = process._linkedBinding('electron_browser_workbuddy_storage');"
    "process.stdout.write(b.loggerGet().toString('utf8'));"
)

_key_lock = threading.Lock()
_cached_key: dict | None = None  # {"key": bytes, "keyId": str}


def _fetch_key_payload() -> dict:
    """通过 WorkBuddy 自带 Electron（run-as-node）调用原生 loggerGet 获取 key payload。"""
    last_err = None
    exes = _electron_candidates()
    if not exes:
        hint = (
            "未找到 WorkBuddy 可执行文件。请用 WORKBUDDY_ELECTRON_PATH 指定其绝对路径"
            "（例如 Windows: C:\\\\Program Files\\\\WorkBuddyAI\\\\WorkBuddyAI.exe）。"
        )
        if sys.platform == "win32":
            hint += " 已扫描：%ProgramFiles%\\WorkBuddyAI、%ProgramFiles%\\WorkBuddy、%LOCALAPPDATA%\\Programs。"
        raise RuntimeError(f"无法获取 WorkBuddy at-rest 密钥（loggerGet）。{hint}")
    for exe in exes:
        env = dict(os.environ, ELECTRON_RUN_AS_NODE="1")
        try:
            r = subprocess.run(
                [str(exe), "-e", _LOGGERGET_JS],
                capture_output=True, text=True, env=env, timeout=20,
            )
            if r.returncode == 0 and r.stdout.strip():
                return json.loads(r.stdout)
            last_err = RuntimeError(f"{exe}: rc={r.returncode} {r.stderr[:200]}")
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(
        "无法获取 WorkBuddy at-rest 密钥（loggerGet）。"
        f"已尝试 {len(exes)} 个可执行文件；也可设置 WORKBUDDY_AT_REST_SECRET "
        f"环境变量直接提供 atRestSecretKey（44 字符 base64）。最后错误：{last_err}"
    )


def protector_key() -> tuple[bytes, str]:
    """返回 (protector_key 32B, keyId 16hex)。结果缓存；失败时按 env 兜底。"""
    global _cached_key
    with _key_lock:
        if _cached_key:
            return _cached_key["key"], _cached_key["keyId"]
        secret_b64 = os.environ.get("WORKBUDDY_AT_REST_SECRET")
        if not secret_b64:
            secret_b64 = _fetch_key_payload()["atRestSecretKey"]
        key = hashlib.sha256(secret_b64.encode("utf-8")).digest()
        key_id = hashlib.sha256(key).hexdigest()[:16]
        _cached_key = {"key": key, "keyId": key_id}
        return key, key_id


# ---------------------------------------------------------------------------
# AAD 构造（sym-v1 field framing）
# ---------------------------------------------------------------------------

def _uint32(v: int) -> bytes:
    return v.to_bytes(4, "big")


def _lp(b: bytes) -> bytes:
    return _uint32(len(b)) + b


def _aad_field(key_id: str) -> bytes:
    return (
        b"WB-AAD\x00"
        + b"\x01"
        + _lp(b"WBEV1")        # field format id
        + _lp(b"sym-v1")       # scheme
        + _uint32(1)           # suite
        + _lp(key_id.encode("ascii"))
        + b"\x02"              # FRAMING_CODE.field
        + b"\x00"              # sequence = undefined
        + b"\x00"              # final = undefined
    )


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------

def is_encrypted_field(value) -> bool:
    return (
        isinstance(value, dict)
        and value.get("$wbEncrypted") == 1
        and isinstance(value.get("envelope"), str)
    )


def decrypt_auth_field(value) -> str:
    """
    解密 .info 中的字段值：
      - 明文字符串（5.6 之前 / 旧版客户端）→ 原样返回；
      - {"$wbEncrypted":1,"envelope":...} → 解密后返回明文字符串。
    """
    if not is_encrypted_field(value):
        return value if isinstance(value, str) else ""
    if AESGCM is None:
        raise RuntimeError("需要 cryptography 库：pip install cryptography")
    key, _ = protector_key()
    env = json.loads(base64.b64decode(value["envelope"]))
    if env.get("suite") != 1:
        raise RuntimeError(f"不支持的加密 suite：{env.get('suite')}")
    key_id = env["keyId"]
    _, cur_key_id = protector_key()
    if key_id != cur_key_id:
        raise RuntimeError(
            f"envelope keyId({key_id}) 与本机派生 keyId({cur_key_id}) 不一致，"
            "WorkBuddy 可能已更换密钥，请重新获取"
        )
    aead = AESGCM(key)
    pt = aead.decrypt(
        base64.b64decode(env["nonce"]),
        base64.b64decode(env["ciphertext"]) + base64.b64decode(env["authTag"]),
        _aad_field(key_id),
    )
    return pt.decode("utf-8")


def encrypt_auth_field(plaintext: str) -> dict:
    """
    将明文加密回 $wbEncrypted 信封（用于刷新 token 后安全回写 .info，
    保持客户端 5.6+ 可读）。nonce 使用 os.urandom(12)。
    """
    if AESGCM is None:
        raise RuntimeError("需要 cryptography 库：pip install cryptography")
    key, key_id = protector_key()
    nonce = os.urandom(12)
    aead = AESGCM(key)
    ct = aead.encrypt(nonce, plaintext.encode("utf-8"), _aad_field(key_id))
    envelope = {
        "suite": 1,
        "keyId": key_id,
        "nonce": base64.b64encode(nonce).decode(),
        "authTag": base64.b64encode(ct[-16:]).decode(),
        "ciphertext": base64.b64encode(ct[:-16]).decode(),
    }
    return {
        "$wbEncrypted": 1,
        "envelope": base64.b64encode(
            json.dumps(envelope, separators=(",", ":")).encode()
        ).decode(),
    }


if __name__ == "__main__":
    # 自检：对本机 .info 做解密往返验证（输出脱敏）
    home = Path.home()
    if sys.platform == "darwin":
        auth_dir = home / "Library/Application Support/CodeBuddyExtension/Data/Public/auth"
    elif sys.platform == "win32":
        local = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
        auth_dir = local / "CodeBuddyExtension/Data/Public/auth"
    else:
        xdg = Path(os.environ.get("XDG_DATA_HOME", home / ".local/share"))
        auth_dir = xdg / "CodeBuddyExtension/Data/Public/auth"
    files = sorted(auth_dir.glob("*.info")) if auth_dir.is_dir() else []
    if not files:
        print(f"未找到 auth 文件（{auth_dir}），跳过自检")
        sys.exit(0)
    print(f"候选可执行文件：{[str(p) for p in _electron_candidates()] or '（无）'}")
    d = json.loads(files[0].read_text())
    tok = d.get("auth", {}).get("accessToken")
    if is_encrypted_field(tok):
        pt = decrypt_auth_field(tok)
        rt = encrypt_auth_field(pt)
        assert decrypt_auth_field(rt) == pt, "加解密往返失败"
        print(f"accessToken 解密成功：JWT 头部={pt[:20]}…，长度 {len(pt)}，往返加密验证通过")
    else:
        print("该 auth 文件为明文格式（旧版客户端），无需处理")
