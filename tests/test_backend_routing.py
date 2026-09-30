"""后端主机按账号 domain 解析（CodeBuddy 与 WorkBuddy 走不同网关）。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.converter import (  # noqa: E402
    BACKEND_DEFAULT,
    CredentialManager,
    resolve_backend,
)


def test_workbuddy_domain_routes_to_workbuddy_host(monkeypatch):
    monkeypatch.delenv("CODEBUDDY_BACKEND", raising=False)
    assert resolve_backend("www.workbuddy.ai") == "https://www.workbuddy.ai"


def test_codebuddy_domain_keeps_copilot_host(monkeypatch):
    monkeypatch.delenv("CODEBUDDY_BACKEND", raising=False)
    assert resolve_backend("www.codebuddy.cn") == BACKEND_DEFAULT


def test_unknown_domain_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("CODEBUDDY_BACKEND", raising=False)
    assert resolve_backend(None) == BACKEND_DEFAULT
    assert resolve_backend("") == BACKEND_DEFAULT
    assert resolve_backend("example.com") == BACKEND_DEFAULT


def test_workbuddy_suffix_heuristic(monkeypatch):
    monkeypatch.delenv("CODEBUDDY_BACKEND", raising=False)
    assert resolve_backend("workbuddy.ai") == "https://www.workbuddy.ai"


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("CODEBUDDY_BACKEND", "https://staging.example.com/")
    assert resolve_backend("www.workbuddy.ai") == "https://staging.example.com"


def test_credential_manager_backend_follows_auth_domain(tmp_path, monkeypatch):
    monkeypatch.delenv("CODEBUDDY_BACKEND", raising=False)
    f = tmp_path / "workbuddy-desktop.info"
    f.write_text(
        json.dumps(
            {
                "auth": {"accessToken": "t", "domain": "www.workbuddy.ai", "expiresAt": 9999999999999},
                "account": {"uid": "u1"},
            }
        ),
        encoding="utf-8",
    )
    cm = CredentialManager(f)
    assert cm.backend() == "https://www.workbuddy.ai"
