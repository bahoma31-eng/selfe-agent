# =============================================================
# agent_ci.py — Selfe Agent v5.4.1 (نسخة GitHub Actions)
# جديد v5.4.1: إصلاح react_loop — منع التوقف عند غياب الأداة
# =============================================================

import os
import sys
import time
import re
import json
import base64
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, TypeVar

MODELS_FILE        = "models.txt"
SYSTEM_PROMPT_FILE = "system_prompt.txt"
MAX_RETRIES        = 3
MAX_MEMORY_TURNS   = 5
MEMORY_DIR         = "memory"

EVAL_THRESHOLD        = 6
MAX_SELF_EVAL_RETRIES = 2
EVAL_LOG_PATH         = f"{MEMORY_DIR}/eval_log.jsonl"
PROMPT_STATS_PATH     = f"{MEMORY_DIR}/prompt_stats.json"
ERROR_LOG_PATH        = f"{MEMORY_DIR}/error_log.jsonl"

PROVIDER_CONFIG = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "secret_vars": [
            "GEMINI_API_KEY_1", "GEMINI_API_KEY_2",
            "GEMINI_API_KEY_3", "GEMINI_API_KEY_4",
        ],
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "secret_vars": ["GROQ_API_KEY"],
    },
}

# ===================================================================
# ErrorMonitor — نظام تتبّع الأخطاء المنظّم (v5.4.0)
# ===================================================================

SEVERITY_INFO     = "INFO"
SEVERITY_WARNING  = "WARNING"
SEVERITY_ERROR    = "ERROR"
SEVERITY_CRITICAL = "CRITICAL"

ERROR_SEVERITY_MAP = {
    400: SEVERITY_ERROR,
    401: SEVERITY_CRITICAL,
    403: SEVERITY_CRITICAL,
    429: SEVERITY_WARNING,
    500: SEVERITY_ERROR,
    503: SEVERITY_WARNING,
}


class ErrorMonitor:
    """\u0646\u0638\u0627\u0645 \u062a\u062a\u0628\u0651\u0639 \u0648\u062a\u0633\u062c\u064a\u0644 \u0627\u0644\u0623\u062e\u0637\u0627\u0621 \u0628\u0635\u064a\u063a\u0629 \u0645\u0646\u0638\u0651\u0645\u0629."""

    def __init__(self, owner: str, repo: str, issue_number: str, model_name: str):
        self.owner        = owner
        self.repo         = repo
        self.issue_number = issue_number
        self.model_name   = model_name
        self._buffer: List[dict] = []

    def _classify(self, error_str: str) -> tuple:
        code_match = re.search(r"Error code:\s*(\d+)", str(error_str))
        code = int(code_match.group(1)) if code_match else 0
        severity = ERROR_SEVERITY_MAP.get(code, SEVERITY_ERROR)
        return code, severity

    def _should_rotate_key(self, error_code: int) -> bool:
        return error_code == 429

    def _should_retry_later(self, error_code: int) -> bool:
        return error_code in (429, 503)

    def _retry_delay(self, error_code: int, attempt: int) -> float:
        if error_code == 429:
            return 20.0
        return 2 ** attempt

    def log(self, error: Exception, context: str = "", step: int = 0) -> dict:
        error_str  = str(error)
        code, severity = self._classify(error_str)
        entry = {
            "ts":           datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "severity":     severity,
            "error_code":   code,
            "model":        self.model_name,
            "issue":        self.issue_number,
            "step":         step,
            "context":      context,
            "message":      error_str[:500],
            "rotate_key":   self._should_rotate_key(code),
            "retry_later":  self._should_retry_later(code),
        }
        icon = {"INFO": "\u2139", "WARNING": "\u26a0", "ERROR": "\u2716", "CRITICAL": "\U0001f534"}.get(severity, "?")
        print(f"[ErrorMonitor] {icon} [{severity}] code={code} ctx={context} \u2192 {error_str[:120]}")
        self._buffer.append(entry)
        return entry

    def flush_to_github(self):
        if not self._buffer:
            return
        try:
            existing = read_file_from_github(self.owner, self.repo, ERROR_LOG_PATH) or ""
            lines    = "\n".join(json.dumps(e, ensure_ascii=False) for e in self._buffer)
            push_file_to_github(
                self.owner, self.repo, ERROR_LOG_PATH,
                existing.rstrip("\n") + "\n" + lines + "\n",
                f"monitor(error-log): {len(self._buffer)} event(s)",
            )
        except Exception as e:
            print(f"[ErrorMonitor] \u26a0 flush failed: {e}")
        finally:
            self._buffer.clear()


# ===================================================================
# SmartAPIClient
# ===================================================================

class SmartAPIClient:
    def __init__(self, keys: List[str], base_url: str, model_name: str, monitor: ErrorMonitor):
        from openai import OpenAI
        self._OpenAI    = OpenAI
        self.keys       = keys
        self.base_url   = base_url
        self.model_name = model_name
        self.monitor    = monitor
        self._key_idx   = 0
        self._client    = self._make_client()

    def _make_client(self):
        from openai import OpenAI
        return OpenAI(api_key=self.keys[self._key_idx], base_url=self.base_url)

    def _rotate_key(self):
        if len(self.keys) <= 1:
            print("[SmartAPIClient] \u26a0 \u0645\u0641\u062a\u0627\u062d \u0648\u0627\u062d\u062f \u0641\u0642\u0637\u060c \u0644\u0627 \u064a\u0648\u062c\u062f \u0645\u0641\u062a\u0627\u062d \u0628\u062f\u064a\u0644.")
            return False
        self._key_idx = (self._key_idx + 1) % len(self.keys)
        self._client  = self._make_client()
        print(f"[SmartAPIClient] \U0001f504 \u062a\u062f\u0648\u064a\u0631 \u0627\u0644\u0645\u0641\u062a\u0627\u062d \u2192 \u0645\u0641\u062a\u0627\u062d #{self._key_idx + 1}")
        return True

    def chat_completions_create(self, step: int = 0, **kwargs) -> Any:
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as e:
                entry = self.monitor.log(e, context="chat_completions_create", step=step)
                last_error = e
                if entry["error_code"] == 429:
                    self._rotate_key()
                    delay = entry.get("retry_delay", 20.0)
                    print(f"[SmartAPIClient] \u23f3 \u0627\u0646\u062a\u0638\u0627\u0631 {delay}s \u0628\u0639\u062f 429...")
                    time.sleep(delay)
                elif entry["error_code"] == 503:
                    delay = 2 ** (attempt + 1)
                    print(f"[SmartAPIClient] \u23f3 \u0627\u0646\u062a\u0638\u0627\u0631 {delay}s \u0628\u0639\u062f 503...")
                    time.sleep(delay)
                elif entry["severity"] == SEVERITY_CRITICAL:
                    raise
                elif attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
        raise last_error

    @property
    def raw(self):
        return self._client


# ===================================================================
# ToolRegistry
# ===================================================================

ToolType = TypeVar('ToolType', bound=Callable[..., Any])

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Any] = {}

    def register_tool(self, tool: Any, name: Optional[str] = None):
        tool_name = name
        if tool_name is None:
            if hasattr(tool, '__name__'):
                tool_name = tool.__name__
            elif hasattr(tool, '__class__') and hasattr(tool.__class__, '__name__'):
                tool_name = tool.__class__.__name__
            else:
                raise ValueError("\u062a\u0639\u0630\u0651\u0631 \u062a\u062d\u062f\u064a\u062f \u0627\u0633\u0645 \u0627\u0644\u0623\u062f\u0627\u0629.")
        if tool_name in self._tools:
            raise ValueError(f"\u0627\u0644\u0623\u062f\u0627\u0629 '{tool_name}' \u0645\u0633\u062c\u0644\u0629 \u0645\u0633\u0628\u0642\u0627\u064b.")
        self._tools[tool_name] = tool

    def get_tool(self, name: str) -> Optional[Any]:
        return self._tools.get(name)

    def list_tool_names(self) -> List[str]:
        return list(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __str__(self) -> str:
        return f"ToolRegistry({', '.join(self.list_tool_names())})"


# ===================================================================
# GitHub API
# ===================================================================

def _github_request(method: str, path: str, data: dict = None) -> dict:
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise RuntimeError("GITHUB_TOKEN \u063a\u064a\u0631 \u0645\u0648\u062c\u0648\u062f.")
    url = f"https://api.github.com{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": "selfe-agent/5.4.1",
    }
    body = json.dumps(data).encode() if data else None
    req  = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"GitHub API {e.code}: {e.read().decode()}")


def get_file_sha(owner, repo, path, branch="main"):
    try:
        return _github_request("GET", f"/repos/{owner}/{repo}/contents/{path}?ref={branch}").get("sha")
    except RuntimeError:
        return None


def push_file_to_github(owner, repo, filepath, content, commit_message, branch="main"):
    existing_sha = get_file_sha(owner, repo, filepath, branch)
    payload = {
        "message": commit_message,
        "content": base64.b64encode(content.encode()).decode(),
        "branch": branch,
    }
    if existing_sha:
        payload["sha"] = existing_sha
    result = _github_request("PUT", f"/repos/{owner}/{repo}/contents/{filepath}", payload)
    return f"https://github.com/{owner}/{repo}/commit/{result.get('commit',{}).get('sha','')}"


def read_file_from_github(owner, repo, filepath, branch="main"):
    try:
        result = _github_request("GET", f"/repos/{owner}/{repo}/contents/{filepath}?ref={branch}")
        return base64.b64decode(result.get("content", "").replace("\n", "")).decode("utf-8")
    except RuntimeError:
        return None


# ===================================================================
# ReAct Loop — v5.4.1
# إصلاح: عند غياب الأداة (action is None) — لا ينتهي اللوب بل يطلب من النموذج المتابعة
# ===================================================================

REACT_SYSTEM_PROMPT = """\
\u0623\u0646\u062a Selfe\u060c \u0648\u0643\u064a\u0644 \u0630\u0643\u0627\u0621 \u0627\u0635\u0637\u0646\u0627\u0639\u064a \u0645\u062a\u0642\u062f\u0651\u0645.
\u0644\u062f\u064a\u0643 \u0635\u0644\u0627\u062d\u064a\u0629 \u0627\u0633\u062a\u062e\u062f\u0627\u0645 \u0627\u0644\u0623\u062f\u0648\u0627\u062a \u0627\u0644\u062a\u0627\u0644\u064a\u0629:

1. push_file   — \u0631\u0641\u0639 \u0645\u0644\u0641 \u0625\u0644\u0649 GitHub
2. read_file   — \u0642\u0631\u0627\u0621\u0629 \u0645\u0644\u0641 \u0645\u0646 GitHub
3. list_files  — \u0639\u0631\u0636 \u0642\u0627\u0626\u0645\u0629 \u0627\u0644\u0645\u0644\u0641\u0627\u062a \u0641\u064a \u0645\u062c\u0644\u062f
4. answer      — \u0625\u0631\u062c\u0627\u0639 \u0627\u0644\u0631\u062f \u0627\u0644\u0646\u0647\u0627\u0626\u064a \u0644\u0644\u0645\u0633\u062a\u062e\u062f\u0645

\u0641\u0648\u0631\u0645\u0627\u062a \u0627\u0633\u062a\u062e\u062f\u0627\u0645 \u0627\u0644\u0623\u062f\u0648\u0627\u062a (\u0627\u0643\u062a\u0628 \u0648\u0627\u062d\u062f\u0629\u064b \u0641\u0642\u0637 \u0641\u064a \u0643\u0644 \u0631\u062f):

\u0644\u0644\u0642\u0631\u0627\u0621\u0629:
```json
{"tool": "read_file", "path": "<\u0645\u0633\u0627\u0631 \u0627\u0644\u0645\u0644\u0641>"}
```
\u0644\u0639\u0631\u0636 \u0627\u0644\u0645\u0644\u0641\u0627\u062a:
```json
{"tool": "list_files", "path": "<\u0627\u0644\u0645\u062c\u0644\u062f>"}
```
\u0644\u0644\u0631\u0641\u0639:
```json
{"tool": "push_file", "path": "<\u0645\u0633\u0627\u0631>", "content": "<\u0627\u0644\u0645\u062d\u062a\u0648\u0649>", "message": "<\u0631\u0633\u0627\u0644\u0629 commit>"}
```
\u0644\u0644\u0625\u062c\u0627\u0628\u0629 \u0627\u0644\u0646\u0647\u0627\u0626\u064a\u0629:
```json
{"tool": "answer", "text": "<\u0631\u062f\u0643 \u0627\u0644\u0646\u0647\u0627\u0626\u064a \u0647\u0646\u0627>"}
```

\u0642\u0648\u0627\u0639\u062f \u0645\u0647\u0645\u0629:
- \u0627\u0643\u062a\u0628 \u062f\u0627\u0626\u0645\u0627 JSON \u062f\u0627\u062e\u0644 ```json ... ``` \u0641\u0642\u0637
- \u0644\u0627 \u062a\u0643\u062a\u0628 \u0623\u062f\u0627\u062a\u064a\u0646 \u0641\u064a \u0646\u0641\u0633 \u0627\u0644\u0631\u062f
- \u0625\u0630\u0627 \u0644\u0645 \u062a\u062d\u062a\u062c \u0623\u062f\u0648\u0627\u062a\u060c \u0627\u0633\u062a\u062e\u062f\u0645 answer \u0645\u0628\u0627\u0634\u0631\u0629
- \u0644\u0627 \u062a\u0636\u0639 \u0623\u064a \u0646\u0635 \u0642\u0628\u0644 JSON \u0623\u0648 \u0628\u0639\u062f\u0647 \u0641\u064a \u0646\u0641\u0633 \u0627\u0644\u0631\u062f
"""

REACT_KEYWORDS = [
    "\u062b\u0645", "\u0628\u0639\u062f \u0630\u0644\u0643", "\u062d\u0644\u0651\u0644", "\u0627\u0642\u0631\u0623", "\u062a\u062d\u0642\u0651\u0642",
    "\u062e\u0637\u0648\u0627\u062a", "\u0639\u062f\u0629", "\u0623\u0648\u0644\u0627\u064b", "\u062b\u0627\u0646\u064a\u0627\u064b", "\u062b\u0627\u0644\u062b\u0627\u064b",
    "\u0627\u0641\u062d\u0635", "\u0639\u062f\u0644", "\u0631\u0627\u062c\u0639",
    "then", "after that", "analyze", "check", "read", "steps", "multiple",
]


def is_complex_task(msg: str) -> bool:
    msg_lower = msg.lower()
    matched = sum(1 for kw in REACT_KEYWORDS if kw in msg_lower)
    return matched >= 2 or (matched >= 1 and len(msg) > 300)


def parse_tool_call(text: str) -> Optional[dict]:
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"(\{[^{}]*\"tool\"[^{}]*\})", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None


def execute_tool(action: dict, owner: str, repo: str) -> str:
    tool = action.get("tool", "")

    if tool == "push_file":
        path    = action.get("path", "output.txt")
        content = action.get("content", "")
        message = action.get("message", f"feat: push {path} via Selfe Agent")
        try:
            commit_url = push_file_to_github(owner, repo, path, content, message)
            return f"\u2705 \u062a\u0645 \u0631\u0641\u0639 `{path}` \u0628\u0646\u062c\u0627\u062d.\ncommit: {commit_url}"
        except Exception as e:
            return f"\u274c \u0641\u0634\u0644 push_file: {e}"

    elif tool == "read_file":
        path = action.get("path", "")
        if not path:
            return "\u274c \u064a\u062c\u0628 \u062a\u062d\u062f\u064a\u062f path."
        content = read_file_from_github(owner, repo, path)
        if content is None:
            return f"\u274c \u0644\u0645 \u064a\u064f\u0639\u062b\u0631 \u0639\u0644\u0649 `{path}`."
        if len(content) > 8000:
            content = content[:8000] + "\n...[truncated]"
        return f"\U0001f4c4 \u0645\u062d\u062a\u0648\u0649 `{path}`:\n```\n{content}\n```"

    elif tool == "list_files":
        path = action.get("path", "")
        api_path = f"/repos/{owner}/{repo}/contents"
        if path and path != "/":
            api_path += f"/{path.strip('/')}"
        try:
            items = _github_request("GET", api_path)
            if not isinstance(items, list):
                return f"\u274c \u0644\u0645 \u064a\u064f\u0639\u062b\u0631 \u0639\u0644\u0649 \u0627\u0644\u0645\u062c\u0644\u062f `{path}`."
            lines = []
            for item in items:
                icon = "\U0001f4c1" if item.get("type") == "dir" else "\U0001f4c4"
                lines.append(f"{icon} {item['name']}")
            return f"\U0001f4c2 \u0645\u062d\u062a\u0648\u064a\u0627\u062a `{'/' if not path else path}`:\n" + "\n".join(lines)
        except Exception as e:
            return f"\u274c \u062e\u0637\u0623 list_files: {e}"

    elif tool == "answer":
        return action.get("text", "")

    else:
        return f"\u26a0\ufe0f \u0623\u062f\u0627\u0629 \u063a\u064a\u0631 \u0645\u0639\u0631\u0648\u0641\u0629: `{tool}`"


def react_loop(
    smart_client: "SmartAPIClient",
    model_name: str,
    messages: list,
    owner: str,
    repo: str,
    max_steps: int = 6,
) -> tuple:
    """
    \u062f\u0648\u0631\u0629 ReAct \u0645\u0639 SmartAPIClient:
      Thought \u2192 Action (JSON) \u2192 Observation \u2192 ... \u2192 answer
    \u062a\u0631\u062c\u0639 (final_reply: str, total_tokens: int)

    v5.4.1: \u0639\u0646\u062f \u063a\u064a\u0627\u0628 \u0627\u0644\u0623\u062f\u0627\u0629 (action is None)\u060c \u0644\u0627 \u064a\u0646\u062a\u0647\u064a \u0627\u0644\u0644\u0648\u0628 \u0628\u0644 \u064a\u0637\u0644\u0628
    \u0645\u0646 \u0627\u0644\u0646\u0645\u0648\u0630\u062c \u062a\u0648\u0636\u064a\u062d \u0645\u0627 \u0625\u0630\u0627 \u0627\u0646\u062a\u0647\u0649 \u0623\u0645 \u064a\u062c\u0628 \u0627\u0644\u0645\u062a\u0627\u0628\u0639\u0629.
    """
    total_tokens = 0
    log_steps    = []

    for step in range(1, max_steps + 1):
        print(f"[ReAct] \u062e\u0637\u0648\u0629 {step}/{max_steps}")

        try:
            resp = smart_client.chat_completions_create(
                step=step,
                model=model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=4096,
            )
            raw    = resp.choices[0].message.content
            tokens = getattr(resp.usage, "total_tokens", 0)
            total_tokens += tokens
        except Exception as e:
            return f"\u26a0\ufe0f \u062e\u0637\u0623 ReAct API (\u062e\u0637\u0648\u0629 {step}): {e}", total_tokens

        print(f"[ReAct] \u0631\u062f \u0627\u0644\u0646\u0645\u0648\u0630\u062c:\n{raw[:300]}...")

        action = parse_tool_call(raw)

        # ============================================================
        # إصلاح v5.4.1: بدل من إنهاء الحلقة، نطلب من النموذج المتابعة
        # ============================================================
        if action is None:
            print(f"[ReAct] \u0644\u0645 \u062a\u064f\u0639\u062b\u0631 \u0639\u0644\u0649 \u0623\u062f\u0627\u0629 \u0641\u064a \u0627\u0644\u062e\u0637\u0648\u0629 {step}\u060c \u0637\u0644\u0628 \u0627\u0644\u062a\u0648\u0636\u064a\u062d.")
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": "\u0644\u0645 \u0623\u062a\u0644\u0642\u064e\u0651 \u0623\u062f\u0627\u0629. \u0647\u0644 \u0627\u0646\u062a\u0647\u064a\u062a \u0645\u0646 \u0627\u0644\u0645\u0647\u0645\u0629\u061f \u0625\u0630\u0627 \u0646\u0639\u0645 \u0627\u0633\u062a\u062e\u062f\u0645 tool=answer\u060c \u0648\u0625\u0644\u0627 \u062a\u0627\u0628\u0639 \u0645\u0639 \u0627\u0644\u0623\u062f\u0627\u0629 \u0627\u0644\u062a\u0627\u0644\u064a\u0629."
            })
            continue  # تابع الحلقة بدل من إنهائها

        tool_name = action.get("tool", "")
        log_steps.append(f"**Step {step}:** tool=`{tool_name}`")

        if tool_name == "answer":
            final = action.get("text", raw)
            return final, total_tokens

        observation = execute_tool(action, owner, repo)
        print(f"[ReAct] Observation: {observation[:200]}")

        messages.append({"role": "assistant",  "content": raw})
        messages.append({"role": "user",       "content": f"Observation: {observation}"})

    # بلغنا max_steps — نطلب رداً نهائياً
    messages.append({"role": "user", "content": "\u0644\u0642\u062f \u0627\u0646\u062a\u0647\u062a \u062c\u0645\u064a\u0639 \u0627\u0644\u062e\u0637\u0648\u0627\u062a. \u0623\u062c\u0628 \u0628\u0634\u0643\u0644 \u0646\u0647\u0627\u0626\u064a \u0628\u0627\u0633\u062a\u062e\u062f\u0627\u0645 tool=answer."})
    try:
        resp = smart_client.chat_completions_create(
            step=max_steps + 1,
            model=model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
        )
        raw    = resp.choices[0].message.content
        total_tokens += getattr(resp.usage, "total_tokens", 0)
        action = parse_tool_call(raw)
        if action and action.get("tool") == "answer":
            return action.get("text", raw), total_tokens
        return raw, total_tokens
    except Exception as e:
        return f"\u26a0\ufe0f \u062e\u0637\u0623 ReAct \u0646\u0647\u0627\u064a\u064a: {e}", total_tokens


# ===================================================================
# MemoryManager
# ===================================================================

class MemoryManager:
    def __init__(self, owner, repo, issue_number):
        self.owner        = owner
        self.repo         = repo
        self.issue_number = issue_number
        self.issue_path   = f"{MEMORY_DIR}/issue_{issue_number}.json"
        self.log_path     = f"{MEMORY_DIR}/global_log.jsonl"
        self._issue_data  = None

    def load_issue_memory(self):
        if self._issue_data is not None:
            return self._issue_data
        raw = read_file_from_github(self.owner, self.repo, self.issue_path)
        if raw:
            try:
                self._issue_data = json.loads(raw)
                print(f"[Memory] issue #{self.issue_number} \u2014 {len(self._issue_data.get('turns',[]))} turn(s)")
            except json.JSONDecodeError:
                self._issue_data = self._empty_issue()
        else:
            self._issue_data = self._empty_issue()
        return self._issue_data

    def _empty_issue(self):
        return {"issue_number": self.issue_number, "created_at": self._now(), "turns": []}

    def build_messages(self, system_prompt, current_msg):
        data    = self.load_issue_memory()
        recent  = data.get("turns", [])[-MAX_MEMORY_TURNS:]
        messages = [{"role": "system", "content": system_prompt}]
        if recent:
            messages[0]["content"] += f"\n\n## \u0633\u064a\u0627\u0642 \u0622\u062e\u0631 {len(recent)} \u062a\u0641\u0627\u0639\u0644:\n"
        for t in recent:
            messages.append({"role": "user",     "content": t["user"]})
            messages.append({"role": "assistant", "content": t["agent"]})
        messages.append({"role": "user", "content": current_msg})
        return messages

    def save_turn(self, user_msg, agent_reply, model_name, tokens, temperature, success):
        data  = self.load_issue_memory()
        turns = data.get("turns", [])
        turns.append({
            "turn": len(turns)+1, "timestamp": self._now(),
            "user": user_msg, "agent": agent_reply,
            "model": model_name, "tokens": tokens,
            "temperature": temperature, "success": success,
        })
        if len(turns) > MAX_MEMORY_TURNS * 2:
            turns = turns[-(MAX_MEMORY_TURNS * 2):]
        data["turns"]      = turns
        data["updated_at"] = self._now()
        self._issue_data   = data
        try:
            push_file_to_github(
                self.owner, self.repo, self.issue_path,
                json.dumps(data, ensure_ascii=False, indent=2),
                f"memory(issue-{self.issue_number}): turn {len(turns)}",
            )
        except Exception as e:
            print(f"[Memory] \u26a0 {e}")
        try:
            existing = read_file_from_github(self.owner, self.repo, self.log_path) or ""
            entry    = json.dumps({"ts": self._now(), "issue": self.issue_number,
                                   "turn": len(turns), "model": model_name,
                                   "tokens": tokens, "success": success}, ensure_ascii=False)
            push_file_to_github(self.owner, self.repo, self.log_path,
                                existing.rstrip("\n") + "\n" + entry + "\n",
                                f"memory(log): issue-{self.issue_number}")
        except Exception as e:
            print(f"[Memory] log \u26a0 {e}")

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ===================================================================
# SelfEvaluator
# ===================================================================

EVAL_SYSTEM_PROMPT = """\u0623\u0646\u062a \u0645\u064f\u0642\u064a\u0651\u0645 \u0645\u0648\u0636\u0648\u0639\u064a. \u0642\u064a\u0651\u0645 \u0627\u0644\u0631\u062f \u0645\u0646 0 \u0625\u0644\u0649 10.
\u0623\u062c\u0628 \u0628\u0640 JSON \u0641\u0642\u0637:
{"score":<0-10>,"issues":[],"improvements":[],"refined_prompt_addition":""}"""


class SelfEvaluator:
    def __init__(self, owner, repo, smart_client: "SmartAPIClient", model_name):
        self.owner        = owner
        self.repo         = repo
        self.smart_client = smart_client
        self.model_name   = model_name

    def evaluate(self, user_msg, agent_reply):
        try:
            resp = self.smart_client.chat_completions_create(
                step=0,
                model=self.model_name,
                messages=[
                    {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                    {"role": "user",   "content": f"\u0627\u0644\u0637\u0644\u0628:\n{user_msg}\n\n\u0627\u0644\u0631\u062f:\n{agent_reply}"},
                ],
                temperature=0.1, max_tokens=256,
            )
            raw = resp.choices[0].message.content.strip()
            m   = re.search(r"\{.*\}", raw, re.DOTALL)
            return json.loads(m.group()) if m else {"score": 10, "issues": [], "improvements": [], "refined_prompt_addition": ""}
        except Exception as e:
            print(f"[SelfEval] \u26a0 {e}")
            return {"score": 10, "issues": [], "improvements": [], "refined_prompt_addition": ""}

    def refine_system_prompt(self, original_prompt, eval_result):
        lines = []
        if eval_result.get("refined_prompt_addition", "").strip():
            lines.append(eval_result["refined_prompt_addition"])
        if eval_result.get("issues"):
            lines.append("\u062a\u062c\u0646\u0651\u0628: " + "\u061b ".join(eval_result["issues"]))
        if not lines:
            return original_prompt
        return original_prompt + "\n\n## \u062a\u0648\u062c\u064a\u0647\u0627\u062a \u062a\u0644\u0642\u0627\u0626\u064a\u0629:\n" + "\n".join(f"- {l}" for l in lines)

    def log_evaluation(self, issue_number, turn, user_msg, score, model_name, attempt, improved):
        entry    = json.dumps({"ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                               "issue": issue_number, "turn": turn, "model": model_name,
                               "score": score, "attempt": attempt, "improved": improved},
                              ensure_ascii=False)
        existing = read_file_from_github(self.owner, self.repo, EVAL_LOG_PATH) or ""
        try:
            push_file_to_github(self.owner, self.repo, EVAL_LOG_PATH,
                                existing.rstrip("\n") + "\n" + entry + "\n",
                                f"eval(log): score-{score}")
        except Exception as e:
            print(f"[SelfEval] \u26a0 log: {e}")

    def update_prompt_stats(self, score, improved):
        raw   = read_file_from_github(self.owner, self.repo, PROMPT_STATS_PATH)
        stats = json.loads(raw) if raw else {
            "total_evals": 0, "total_score": 0,
            "improvements_triggered": 0, "avg_score": 0.0,
            "score_distribution": {str(i): 0 for i in range(11)},
        }
        stats["total_evals"]  += 1
        stats["total_score"]  += score
        stats["avg_score"]     = round(stats["total_score"] / stats["total_evals"], 2)
        stats["score_distribution"][str(min(score, 10))] = stats["score_distribution"].get(str(min(score, 10)), 0) + 1
        if improved:
            stats["improvements_triggered"] += 1
        stats["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            push_file_to_github(self.owner, self.repo, PROMPT_STATS_PATH,
                                json.dumps(stats, ensure_ascii=False, indent=2),
                                f"eval(stats): avg={stats['avg_score']}")
        except Exception as e:
            print(f"[SelfEval] \u26a0 stats: {e}")


# ===================================================================
# كشف أمر /push
# ===================================================================

PUSH_PATTERN       = re.compile(r"^/push\s+(.+)$", re.IGNORECASE | re.MULTILINE)
FILE_NAME_PATTERN  = re.compile(r"\b([\w\-/]+\.\w+)\b")
CODE_BLOCK_PATTERN = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)


def detect_push_command(message):
    m = PUSH_PATTERN.search(message)
    return (True, m.group(1).strip()) if m else (False, "")


def extract_filename(instruction):
    m = FILE_NAME_PATTERN.search(instruction)
    if m:
        return m.group(1)
    words    = re.sub(r"[^\w\s]", "", instruction).split()
    slug     = "_".join(words[:4]).lower() if words else "script"
    lang_map = {"python": ".py", "py": ".py", "javascript": ".js", "js": ".js",
                "typescript": ".ts", "ts": ".ts", "bash": ".sh", "shell": ".sh",
                "html": ".html", "css": ".css", "yaml": ".yml", "json": ".json"}
    for kw, ext in lang_map.items():
        if kw in instruction.lower():
            return f"{slug}{ext}"
    return f"{slug}.py"


def extract_code_from_reply(reply):
    blocks = CODE_BLOCK_PATTERN.findall(reply)
    return blocks[0].strip() if blocks else reply.strip()


# ===================================================================
# دوال مشتركة
# ===================================================================

def load_system_prompt(filepath):
    default = "\u0623\u0646\u062a Selfe\u060c \u0648\u0643\u064a\u0644 \u0630\u0643\u0627\u0621 \u0627\u0635\u0637\u0646\u0627\u0639\u064a \u0645\u062a\u062e\u0635\u0635 \u0641\u064a \u062a\u0637\u0648\u064a\u0631 \u0627\u0644\u0628\u0631\u0645\u062c\u064a\u0627\u062a."
    if not os.path.exists(filepath):
        return default
    lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip().startswith("#"):
                lines.append(line.rstrip())
    return "\n".join(lines).strip() or default


def load_models(filepath):
    if not os.path.exists(filepath):
        print(f"[ERROR] \u0644\u0645 \u064a\u064f\u0639\u062b\u0631 \u0639\u0644\u0649: {filepath}")
        sys.exit(1)
    models = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, provider = ([p.strip() for p in line.split("|", 1)]
                              if "|" in line else (line, "gemini"))
            if provider.lower() in PROVIDER_CONFIG:
                models.append({"name": name, "provider": provider.lower()})
    if not models:
        print("[ERROR] \u0644\u0627 \u062a\u0648\u062c\u062f \u0646\u0645\u0627\u0630\u062c")
        sys.exit(1)
    return models


def detect_temperature(message):
    msg = message.lower()
    if any(k in msg for k in ["\u0643\u0648\u062f","code","\u062f\u0627\u0644\u0629","function","\u0627\u0643\u062a\u0628","\u0628\u0631\u0645\u062c\u0629","debug","script","/push"]):
        return 0.2
    if any(k in msg for k in ["\u0642\u0635\u064a\u062f\u0629","poem","\u0627\u0642\u062a\u0631\u0627\u062d","\u0641\u0643\u0631\u0629","\u0625\u0628\u062f\u0627\u0639","\u062a\u0635\u0645\u064a\u0645","\u0642\u0635\u0629"]):
        return 0.9
    if any(k in msg for k in ["\u0645\u0627 \u0647\u0648","what is","\u0627\u0634\u0631\u062d","explain","\u0641\u0631\u0642","difference"]):
        return 0.5
    return 0.7


def write_output(reply):
    gh_output = os.environ.get("GITHUB_OUTPUT", "")
    if gh_output:
        with open(gh_output, "a", encoding="utf-8") as fh:
            fh.write(f"reply<<EOF\n{reply}\nEOF\n")
    print("[Selfe Reply]")
    print(reply)


PUSH_SYSTEM_PROMPT = """\u0623\u0646\u062a Selfe\u060c \u0648\u0643\u064a\u0644 \u0628\u0631\u0645\u062c\u0629.
\u0627\u0643\u062a\u0628 \u0627\u0644\u0643\u0648\u062f \u0641\u0642\u0637 \u062f\u0627\u062e\u0644 code fence:
```python
# \u0627\u0644\u0643\u0648\u062f \u0647\u0646\u0627
```"""


# ===================================================================
# main
# ===================================================================

def main():
    print("\n[Selfe Agent CI v5.4.1] \u062a\u0634\u063a\u064a\u0644...")

    msg = os.environ.get("USER_MESSAGE", "").strip()
    if not msg:
        write_output("\u26a0\ufe0f \u0644\u0645 \u064a\u062a\u0645 \u0627\u0633\u062a\u0642\u0628\u0627\u0644 \u0623\u064a \u0631\u0633\u0627\u0644\u0629.")
        sys.exit(0)

    gh_repo      = os.environ.get("GITHUB_REPOSITORY", "/")
    owner, repo  = gh_repo.split("/", 1)
    issue_number = os.environ.get("ISSUE_NUMBER", "0")

    models = load_models(MODELS_FILE)
    idx    = int(os.environ.get("MODEL_INDEX", "1")) - 1
    model  = models[idx] if idx < len(models) else models[0]
    print(f"[CI] \u0627\u0644\u0646\u0645\u0648\u0630\u062c: {model['name']} | {model['provider']} | Issue #{issue_number}")

    all_keys = {
        p: [os.environ.get(v, "").strip() for v in cfg["secret_vars"] if os.environ.get(v, "").strip()]
        for p, cfg in PROVIDER_CONFIG.items()
    }
    pkeys = all_keys.get(model["provider"], [])
    if not pkeys:
        write_output(f"\u26a0\ufe0f \u0644\u0627 \u064a\u0648\u062c\u062f \u0645\u0641\u062a\u0627\u062d API \u0644\u0640: {model['provider']}")
        sys.exit(1)

    memory = MemoryManager(owner, repo, issue_number)

    try:
        from openai import OpenAI
    except ImportError:
        write_output("\u26a0\ufe0f \u0645\u0643\u062a\u0628\u0629 openai \u063a\u064a\u0631 \u0645\u062b\u0628\u062a\u064e\u0651\u062a\u0629.")
        sys.exit(1)

    cfg     = PROVIDER_CONFIG[model["provider"]]

    monitor      = ErrorMonitor(owner, repo, issue_number, model["name"])
    smart_client = SmartAPIClient(pkeys, cfg["base_url"], model["name"], monitor)
    evaluator    = SelfEvaluator(owner, repo, smart_client, model["name"])

    # ── /push ─────────────────────────────────────────────────────
    is_push, push_instruction = detect_push_command(msg)
    if is_push:
        print(f"[CI] /push \u2014 {push_instruction}")
        filename = extract_filename(push_instruction)
        for attempt in range(MAX_RETRIES):
            try:
                resp = smart_client.chat_completions_create(
                    step=0,
                    model=model["name"],
                    messages=[
                        {"role": "system", "content": PUSH_SYSTEM_PROMPT},
                        {"role": "user",   "content": push_instruction},
                    ],
                    temperature=0.2, max_tokens=4096,
                )
                raw_reply  = resp.choices[0].message.content
                clean_code = extract_code_from_reply(raw_reply)
                tokens     = getattr(resp.usage, "total_tokens", 0)
                commit_url = push_file_to_github(
                    owner, repo, filename, clean_code,
                    f"feat({filename}): generated by Selfe Agent via /push",
                )
                reply = (
                    f"\u2705 **\u062a\u0645 \u062f\u0641\u0639 \u0627\u0644\u0645\u0644\u0641 \u0628\u0646\u062c\u0627\u062d!**\n\n"
                    f"\U0001f4c4 **\u0627\u0644\u0645\u0644\u0641:** `{filename}`\n"
                    f"\U0001f517 **\u0627\u0644\u0640 commit:** {commit_url}\n\n"
                    f"```\n{clean_code[:1500]}\n```"
                )
                write_output(reply)
                memory.save_turn(msg, reply, model["name"], tokens, 0.2, True)
                monitor.flush_to_github()
                return
            except Exception as e:
                if attempt >= MAX_RETRIES - 1:
                    monitor.flush_to_github()
                    write_output(f"\u26a0\ufe0f \u062e\u0637\u0623 /push: {e}")
                    sys.exit(1)

    # ── ReAct Loop (مهام معقدة) + SelfEval + Memory ──────
    if is_complex_task(msg):
        print(f"[CI] \u2728 \u0645\u0647\u0645\u0629 \u0645\u0639\u0642\u062f\u0629 \u2192 \u062a\u0641\u0639\u064a\u0644 ReAct Loop")
        base_sp      = load_system_prompt(SYSTEM_PROMPT_FILE)
        data         = memory.load_issue_memory()
        current_turn = len(data.get("turns", [])) + 1

        messages = memory.build_messages(REACT_SYSTEM_PROMPT + "\n\n" + base_sp, msg)
        final_reply, total_tokens = react_loop(
            smart_client, model["name"], messages, owner, repo, max_steps=6
        )

        eval_result = evaluator.evaluate(msg, final_reply)
        score       = eval_result.get("score", 10)
        print(f"[SelfEval/ReAct] {score}/10")

        if score < EVAL_THRESHOLD:
            print(f"[SelfEval/ReAct] \u26a0 score={score} < {EVAL_THRESHOLD} \u2192 \u0625\u0639\u0627\u062f\u0629 \u0645\u062d\u0627\u0648\u0644\u0629 \u0628\u0640 refined prompt")
            refined_sp = evaluator.refine_system_prompt(
                REACT_SYSTEM_PROMPT + "\n\n" + base_sp, eval_result
            )
            messages2 = memory.build_messages(refined_sp, msg)
            retry_reply, retry_tokens = react_loop(
                smart_client, model["name"], messages2, owner, repo, max_steps=6
            )
            eval_result2 = evaluator.evaluate(msg, retry_reply)
            score2       = eval_result2.get("score", 10)
            print(f"[SelfEval/ReAct] retry \u2192 {score2}/10")

            if score2 > score:
                final_reply   = retry_reply
                total_tokens += retry_tokens
                score         = score2

        evaluator.log_evaluation(
            issue_number, current_turn, msg, score,
            model["name"], attempt=0, improved=(score < EVAL_THRESHOLD)
        )
        evaluator.update_prompt_stats(score, improved=(score < EVAL_THRESHOLD))

        if score < EVAL_THRESHOLD:
            final_reply += f"\n\n---\n> \u26a0\ufe0f *\u0623\u0641\u0636\u0644 \u062a\u0642\u064a\u064a\u0645 \u0630\u0627\u062a\u064a: {score}/10*"

        write_output(final_reply)
        memory.save_turn(
            msg, final_reply, model["name"], total_tokens, 0.2,
            success=(score >= EVAL_THRESHOLD)
        )
        monitor.flush_to_github()
        print(f"[CI] ReAct \u0627\u0643\u062a\u0645\u0644 \u2714  (score={score})")
        return

    # ── وضع عادي (Single-shot) + SelfEval + Memory ────
    base_system_prompt    = load_system_prompt(SYSTEM_PROMPT_FILE)
    temp                  = detect_temperature(msg)
    current_system_prompt = base_system_prompt
    final_reply           = None
    best_score            = 0
    best_reply            = None
    final_tokens          = 0

    data         = memory.load_issue_memory()
    current_turn = len(data.get("turns", [])) + 1

    for eval_attempt in range(MAX_SELF_EVAL_RETRIES + 1):
        if eval_attempt > 0:
            print(f"[SelfEval] \U0001f504 \u0625\u0639\u0627\u062f\u0629 \u0645\u062d\u0627\u0648\u0644\u0629 {eval_attempt}...")

        messages = memory.build_messages(current_system_prompt, msg)
        reply    = None
        tokens   = 0

        try:
            resp = smart_client.chat_completions_create(
                step=eval_attempt,
                model=model["name"],
                messages=messages,
                temperature=temp,
                max_tokens=4096,
            )
            reply  = resp.choices[0].message.content
            tokens = getattr(resp.usage, "total_tokens", 0)
        except Exception as e:
            monitor.flush_to_github()
            write_output(f"\u26a0\ufe0f \u062e\u0637\u0623: {e}")
            memory.save_turn(msg, str(e), model["name"], 0, temp, False)
            sys.exit(1)

        if reply is None:
            break

        eval_result = evaluator.evaluate(msg, reply)
        score       = eval_result.get("score", 10)
        print(f"[SelfEval] {score}/10")

        if score > best_score:
            best_score   = score
            best_reply   = reply
            final_tokens = tokens

        evaluator.log_evaluation(issue_number, current_turn, msg, score, model["name"], eval_attempt, eval_attempt > 0)
        evaluator.update_prompt_stats(score, eval_attempt > 0)

        if score >= EVAL_THRESHOLD:
            final_reply = reply
            break
        elif eval_attempt < MAX_SELF_EVAL_RETRIES:
            current_system_prompt = evaluator.refine_system_prompt(base_system_prompt, eval_result)
        else:
            final_reply = best_reply

    if not final_reply:
        final_reply = best_reply or "\u26a0\ufe0f \u0641\u0634\u0644 \u0627\u0644\u0648\u0643\u064a\u0644."

    if best_score < EVAL_THRESHOLD:
        final_reply += f"\n\n---\n> \u26a0\ufe0f *\u0623\u0641\u0636\u0644 \u062a\u0642\u064a\u064a\u0645 \u0630\u0627\u062a\u064a: {best_score}/10*"

    write_output(final_reply)
    memory.save_turn(msg, final_reply, model["name"], final_tokens, temp, True)
    monitor.flush_to_github()
    print("[CI] \u0627\u0643\u062a\u0645\u0644 \u2714")


if __name__ == "__main__":
    main()
