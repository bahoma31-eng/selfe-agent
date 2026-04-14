# =============================================================
# agent_ci.py — Selfe Agent v6.5.0
# إصلاح شامل: فصل طبقات الفشل + Retry ذكي + ReAct محسّن
# v6.1.1: إصلاح TypeError عند raw=None (choices[0].message.content)
# v6.1.0: 3 حلول لاستنزاف حصة API:
#   1. كشف quota_exceeded وتخطي المفتاح كلياً
#   2. تدوير تلقائي بين مزودين (Gemini → Groq) عند 429/quota
#   3. وضع Fallback: Groq كامل عند فشل Gemini بالكامل
# v6.2.0: تحسين REACT_SYSTEM_PROMPT بقاعدة "لا تصف — افعل"
#         + تحسين رسالة الخطأ في react_loop عند غياب الأداة
# v6.3.0: مبدأ الاكتفاء الذاتي — الوكيل يكتب سكريبتاته بنفسه
#         ويستعمل المفاتيح السرية من GitHub Secrets مباشرةً
#         دون الحاجة لأدوات مُعرَّفة مسبقاً (إرسال بريد، نشر فيسبوك...)
# v6.4.0: إصلاح هلوسة الأدوات:
#   1. توحيد اسم المتغير SMTP_PASSWORD → SMTP_PASS
#   2. إضافة validate_execution_result للتحقق من نجاح حقيقي
#   3. إضافة check_required_env قبل تشغيل السكريبتات الخارجية
#   4. تصليح parse_tool_call: رفض JSON خارج code fence
# v6.5.0: ثلاثة تحسينات:
#   1. إضافة أداة send_email مباشرة في execute_tool
#   2. إضافة send_email لجدول الأدوات في REACT_SYSTEM_PROMPT + مثال
#   3. تخفيف parse_tool_call: قبول JSON خارج code fence كخطة بديلة
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
            "GEMINI_API_KEY_5", "GEMINI_API_KEY_6",
            "GEMINI_API_KEY_7", "GEMINI_API_KEY_8",
            "GEMINI_API_KEY_9",
        ],
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "secret_vars": ["GROQ_API_KEY"],
    },
}

# نموذج Groq الاحتياطي المستخدم عند الـ Fallback الكامل
GROQ_FALLBACK_MODEL = "llama-3.3-70b-versatile"

# عبارات تدل على استنزاف الحصة كلياً (ليس مجرد rate limit مؤقت)
QUOTA_EXHAUSTED_PATTERNS = [
    "quota exceeded",
    "quota_exceeded",
    "resource_exhausted",
    "you exceeded your current quota",
    "billing",
    "insufficient_quota",
    "rate limit reached for requests per day",
    "daily limit",
    "monthly limit",
]

# ===================================================================
# قاموس المتغيرات البيئية المطلوبة لكل خدمة خارجية — v6.4.0
# ملاحظة: SMTP_PASS (لا SMTP_PASSWORD) ليتوافق مع GitHub Secrets
# ===================================================================

REQUIRED_ENV_VARS = {
    "email":    ["SMTP_USER", "SMTP_PASS"],
    "facebook": ["FB_PAGE_ACCESS_TOKEN", "FB_PAGE_ID"],
    "telegram": ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"],
    "twilio":   ["TWILIO_AUTH_TOKEN", "TWILIO_ACCOUNT_SID"],
    "imgbb":    ["IMGBB_API_KEY"],
}

SERVICE_KEYWORDS = {
    "email":    ["smtp", "sendmail", "smtplib", "send_message", "mime"],
    "facebook": ["graph.facebook", "fb_page", "fb_access"],
    "telegram": ["api.telegram", "telegram_bot"],
    "twilio":   ["twilio"],
    "imgbb":    ["imgbb", "api.imgbb"],
}


def check_required_env(service: str):
    """يتحقق من وجود المتغيرات البيئية للخدمة المطلوبة."""
    required = REQUIRED_ENV_VARS.get(service.lower(), [])
    missing = [v for v in required if not os.environ.get(v, "").strip()]
    if missing:
        return False, (
            f"❌ متغيرات بيئية مفقودة لخدمة '{service}': {', '.join(missing)}\n"
            f"أضفها كـ GitHub Secrets ثم أعد تشغيل المهمة."
        )
    return True, ""


def detect_service_in_code(code: str) -> Optional[str]:
    """يكشف الخدمة الخارجية المستهدفة من كود السكريبت."""
    code_lower = code.lower()
    for service, keywords in SERVICE_KEYWORDS.items():
        if any(kw in code_lower for kw in keywords):
            return service
    return None


def validate_execution_result(output: str, action: dict) -> str:
    """
    يتحقق من أن نتيجة التنفيذ تدل على نجاح حقيقي، لا مجرد غياب خطأ.
    """
    code = action.get("code", "")
    service = detect_service_in_code(code)

    # إذا كان الإخراج يحتوي على ✅ لكن قصير جداً → مشبوه
    if "✅" in output and len(output.strip()) < 35:
        return (
            "⚠️ تحذير: يبدو أن السكريبت نجح لكن لم يُنتج دليلاً واضحاً على الإنجاز.\n"
            f"{output}\n"
            "يُرجى التحقق من المتغيرات البيئية المطلوبة وإعادة المحاولة."
        )

    # إذا كان مرتبطاً بخدمة → تحقق من المتغيرات قبل الإعلان عن النجاح
    if service:
        ok, err_msg = check_required_env(service)
        if not ok:
            return err_msg

    return output


# ===================================================================
# استراتيجية Retry المخصصة لكل كود HTTP
# ===================================================================

RETRY_STRATEGY: Dict[Any, dict] = {
    429: {"max_attempts": 5, "base_delay": 30.0, "backoff": "linear",      "rotate_key": True},
    503: {"max_attempts": 4, "base_delay": 5.0,  "backoff": "exponential", "rotate_key": False},
    500: {"max_attempts": 3, "base_delay": 2.0,  "backoff": "exponential", "rotate_key": False},
    "default": {"max_attempts": 3, "base_delay": 1.0, "backoff": "fixed",  "rotate_key": False},
}


def _compute_delay(strategy: dict, attempt: int) -> float:
    base = strategy["base_delay"]
    mode = strategy["backoff"]
    if mode == "exponential":
        return base * (2 ** attempt)
    if mode == "linear":
        return base * (attempt + 1)
    return base  # fixed


def _is_quota_exhausted(error_str: str) -> bool:
    low = error_str.lower()
    return any(p in low for p in QUOTA_EXHAUSTED_PATTERNS)


# ===================================================================
# ErrorMonitor
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

    def log(self, error: Exception, context: str = "", step: int = 0) -> dict:
        error_str      = str(error)
        code, severity = self._classify(error_str)
        strategy       = RETRY_STRATEGY.get(code, RETRY_STRATEGY["default"])
        quota_exhausted = _is_quota_exhausted(error_str)
        entry = {
            "ts":             datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "severity":       severity,
            "error_code":     code,
            "model":          self.model_name,
            "issue":          self.issue_number,
            "step":           step,
            "context":        context,
            "message":        error_str[:500],
            "rotate_key":     strategy["rotate_key"],
            "retry_later":    code in (429, 503),
            "max_attempts":   strategy["max_attempts"],
            "quota_exhausted": quota_exhausted,
        }
        icon = {"INFO": "ℹ", "WARNING": "⚠", "ERROR": "✖", "CRITICAL": "🔴"}.get(severity, "?")
        quota_tag = " [QUOTA_EXHAUSTED]" if quota_exhausted else ""
        print(f"[ErrorMonitor] {icon} [{severity}]{quota_tag} code={code} ctx={context} → {error_str[:120]}")
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
            print(f"[ErrorMonitor] ⚠ flush failed: {e}")
        finally:
            self._buffer.clear()


# ===================================================================
# SmartAPIClient — v6.4.0
# ===================================================================

class SmartAPIClient:
    def __init__(
        self,
        keys: List[str],
        base_url: str,
        model_name: str,
        monitor: "ErrorMonitor",
        fallback_keys: Optional[List[str]] = None,
        fallback_base_url: Optional[str] = None,
        fallback_model: Optional[str] = None,
    ):
        from openai import OpenAI
        self._OpenAI          = OpenAI
        self.keys             = keys
        self.base_url         = base_url
        self.model_name       = model_name
        self.monitor          = monitor
        self._key_idx         = 0
        self._exhausted_keys: set = set()
        self.fallback_keys      = fallback_keys or []
        self.fallback_base_url  = fallback_base_url
        self.fallback_model     = fallback_model or GROQ_FALLBACK_MODEL
        self._using_fallback    = False
        self._client = self._make_client()

    def _make_client(self):
        from openai import OpenAI
        if self._using_fallback:
            return self._OpenAI(
                api_key=self.fallback_keys[0],
                base_url=self.fallback_base_url,
            )
        return self._OpenAI(api_key=self.keys[self._key_idx], base_url=self.base_url)

    def _rotate_key(self) -> bool:
        available = [i for i in range(len(self.keys)) if i not in self._exhausted_keys]
        if not available:
            print("[SmartAPIClient] 🔴 جميع المفاتيح مستنزفة.")
            return False
        next_keys = [i for i in available if i != self._key_idx]
        if not next_keys:
            return False
        self._key_idx = next_keys[0]
        self._client  = self._make_client()
        print(f"[SmartAPIClient] 🔄 تدوير المفتاح → مفتاح #{self._key_idx + 1}")
        return True

    def _activate_fallback(self) -> bool:
        if not self.fallback_keys or not self.fallback_base_url:
            print("[SmartAPIClient] ⚠ لا يوجد مزود fallback مُعرَّف.")
            return False
        if self._using_fallback:
            return False
        self._using_fallback = True
        self._client = self._make_client()
        print(
            f"[SmartAPIClient] 🔀 تفعيل Fallback → {self.fallback_model} "
            f"(Groq) بعد استنزاف جميع مفاتيح Gemini"
        )
        return True

    @property
    def active_model(self) -> str:
        return self.fallback_model if self._using_fallback else self.model_name

    def chat_completions_create(self, step: int = 0, **kwargs) -> Any:
        last_error = None
        attempt    = 0
        kwargs["model"] = self.active_model

        while True:
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as e:
                entry      = self.monitor.log(e, context="chat_completions_create", step=step)
                last_error = e
                code       = entry["error_code"]
                severity   = entry["severity"]
                quota_ex   = entry.get("quota_exhausted", False)

                if severity == SEVERITY_CRITICAL:
                    raise

                if quota_ex and not self._using_fallback:
                    print(f"[SmartAPIClient] 🚫 مفتاح #{self._key_idx + 1} مستنزف كلياً — تخطٍّ فوري.")
                    self._exhausted_keys.add(self._key_idx)
                    rotated = self._rotate_key()
                    if not rotated:
                        activated = self._activate_fallback()
                        if activated:
                            kwargs["model"] = self.active_model
                            attempt = 0
                            continue
                        raise
                    kwargs["model"] = self.active_model
                    attempt = 0
                    continue

                strategy     = RETRY_STRATEGY.get(code, RETRY_STRATEGY["default"])
                max_attempts = strategy["max_attempts"]

                if attempt >= max_attempts - 1:
                    if not self._using_fallback:
                        activated = self._activate_fallback()
                        if activated:
                            kwargs["model"] = self.active_model
                            attempt = 0
                            continue
                    break

                if strategy["rotate_key"] and not self._using_fallback:
                    self._rotate_key()
                    kwargs["model"] = self.active_model

                delay = _compute_delay(strategy, attempt)
                print(f"[SmartAPIClient] ⏳ محاولة {attempt+1}/{max_attempts} — انتظار {delay:.1f}s (code={code})")
                time.sleep(delay)
                attempt += 1

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
                raise ValueError("تعذّر تحديد اسم الأداة.")
        if tool_name in self._tools:
            raise ValueError(f"الأداة '{tool_name}' مسجلة مسبقاً.")
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
        raise RuntimeError("GITHUB_TOKEN غير موجود.")
    url = f"https://api.github.com{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": "selfe-agent/6.5.0",
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
# ReAct Loop — v6.5.0
# ===================================================================

REACT_SYSTEM_PROMPT = """\
أنت Selfe، وكيل ذكاء اصطناعي متقدّم يعمل بنمط ReAct (Reason + Act).

## الأدوات المتاحة

| الأداة         | الوصف                                                    |
|---------------|----------------------------------------------------------|
| read_file     | قراءة محتوى ملف من GitHub                                |
| push_file     | رفع/تعديل ملف في GitHub                                  |
| list_files    | عرض الملفات والمجلدات                                    |
| search_files  | البحث عن ملف بالاسم في المستودع                          |
| run_script    | كتابة وتنفيذ سكريبت Python مباشرةً داخل بيئة التشغيل    |
| send_email    | إرسال بريد إلكتروني مباشرةً                              |
| answer        | إرجاع الرد النهائي للمستخدم                              |

## صيغة الاستخدام (إلزامية)

كل رد يجب أن يحتوي على أداة واحدة فقط بهذا الشكل الدقيق:

```json
{"tool": "read_file", "path": "agent_ci.py"}
```

```json
{"tool": "list_files", "path": "memory"}
```

```json
{"tool": "push_file", "path": "output.py", "content": "# code", "message": "feat: add output"}
```

```json
{"tool": "search_files", "query": "agent"}
```

```json
{"tool": "run_script", "code": "import os\nprint(os.environ.get('GITHUB_TOKEN',''))"}
```

```json
{"tool": "send_email", "to": "bahoma31@gmail.com", "subject": "تقرير", "body": "مرحباً..."}
```

```json
{"tool": "answer", "text": "ردك النهائي هنا"}
```

## مبدأ الاكتفاء الذاتي — القاعدة الأساسية

إذا طُلب منك تنفيذ عملية لا تملك أداةً مُعرَّفة لها مسبقاً
(مثل: النشر على فيسبوك، إرسال رسالة Telegram،
استدعاء API خارجي، إرسال SMS، جدولة مهمة...)

**لا تقل أنك عاجز. بل:**

1. **اكتب سكريبت Python** يُنجز المهمة كاملاً
2. **استعمل المفاتيح السرية** المتاحة في بيئة GitHub Actions
   عبر: `os.environ.get("VARIABLE_NAME", "")`
3. **استخدم أداة run_script** لتنفيذ السكريبت فوراً
4. **ارفع السكريبت** كملف في المستودع لإعادة استخدامه لاحقاً
5. **سجّل النتيجة** في تقرير أو تعليق على الـ Issue

### المفاتيح السرية المتاحة دائماً في بيئة التشغيل

| المفتاح               | الاستخدام                              |
|----------------------|----------------------------------------|
| GITHUB_TOKEN         | GitHub API — متاح دائماً              |
| GEMINI_API_KEY_1..9  | نماذج Gemini                           |
| GROQ_API_KEY         | نماذج Groq / LLaMA                    |
| SMTP_PASS            | إرسال البريد الإلكتروني               |
| SMTP_USER            | حساب البريد المُرسِل                  |
| OWNER_EMAIL          | بريد صاحب المستودع                    |
| FB_PAGE_ACCESS_TOKEN | Facebook Graph API                    |
| FB_PAGE_ID           | معرّف صفحة فيسبوك                     |
| IMGBB_API_KEY        | رفع الصور على ImgBB                   |
| IG_USER_ID           | Instagram User ID                     |
| PAT_TOKEN            | GitHub Personal Access Token          |

### أمثلة تطبيقية — ماذا يكتب الوكيل بنفسه

**مثال 1 — إرسال بريد إلكتروني عبر أداة send_email:**
```json
{"tool": "send_email", "to": "user@example.com", "subject": "تقرير من Selfe Agent", "body": "مرحباً، هذا تقريرك اليومي."}
```

**مثال 2 — النشر على فيسبوك:**
```python
import urllib.request, urllib.parse, os, json
token   = os.environ.get("FB_PAGE_ACCESS_TOKEN", "")
page_id = os.environ.get("FB_PAGE_ID", "")
if not token or not page_id:
    print("❌ FB_PAGE_ACCESS_TOKEN أو FB_PAGE_ID غير موجودَين")
else:
    data = urllib.parse.urlencode({"message": "منشور جديد!", "access_token": token}).encode()
    req  = urllib.request.Request(f"https://graph.facebook.com/v19.0/{page_id}/feed", data=data)
    res  = json.loads(urllib.request.urlopen(req).read())
    print(f"✅ نُشر بنجاح — id: {res.get('id')}")
```

## ⚠️ قاعدة التحقق من النجاح (إلزامية) — v6.4.0

بعد كل run_script يُرسل بيانات لخدمة خارجية:
- إذا ظهر "❌" في الـ Observation → المهمة فاشلة، لا تُعلن النجاح، بلّغ المستخدم بالسبب الحقيقي
- إذا ظهر "متغيرات بيئية مفقودة" → أوضح للمستخدم ما يجب إضافته في GitHub Secrets
- لا تُعلن النجاح إلا إذا ظهر "✅" مع دليل واضح (بريد مُرسَل من X، message_id، post_id... إلخ)
- إذا كانت المخرجات فارغة أو مشبوهة → أعد المحاولة أو أبلغ المستخدم

## قواعد صارمة — لا استثناء

1. **أداة واحدة فقط** في كل رد. لا أداتين معاً أبداً.
2. **JSON داخل ```json ... ```** دائماً — يُفضَّل دائماً استخدام code fence.
3. **لا تتوقف قبل الإجابة النهائية** — استخدم tool=answer فقط عندما تنتهي من جميع خطوات المهمة.
4. **إذا فشلت أداة** — انتقل لأداة بديلة أو غيّر المسار، ولا تكرر نفس الخطأ.
5. **لا تكتب أي نص** قبل JSON أو بعده في نفس الرد.
6. **استمر في العمل** حتى تُنجز المهمة كاملةً أو تستنفد جميع الأدوات.
7. **لا تدّعي العجز** — إذا لم تجد أداة مناسبة، اكتب السكريبت بنفسك عبر run_script.
8. **لا تُعلن النجاح بدون دليل** من الـ Observation يثبت اكتمال المهمة.

## ⛔ القاعدة الذهبية: لا تصف ما ستفعله — افعله مباشرةً

الخطأ الشائع (محظور تماماً):
  "سأقوم أولاً بـ X ثم Y، هل توافق؟"
  "سأنفذ الخطوات التالية: 1. ... 2. ..."
  "لا أستطيع إرسال البريد لأنني لا أملك الأداة..."

الصحيح: ابدأ فوراً بأداة JSON دون مقدمات.
"""

REACT_KEYWORDS = [
    "ثم", "بعد ذلك", "حلّل", "اقرأ", "تحقّق",
    "خطوات", "عدة", "أولاً", "ثانياً", "ثالثاً",
    "افحص", "عدل", "راجع", "قارن", "ابحث", "حدّث",
    "then", "after that", "analyze", "check", "read", "steps", "multiple",
    "update", "modify", "compare", "search", "find",
]


def is_complex_task(msg: str) -> bool:
    msg_lower = msg.lower()
    matched = sum(1 for kw in REACT_KEYWORDS if kw in msg_lower)
    return matched >= 1 or len(msg) > 200


def parse_tool_call(text: str) -> Optional[dict]:
    """
    v6.5.0: يقبل JSON داخل ```json ... ``` أولاً (الطريقة المفضّلة).
    إذا لم يُعثر عليه، يبحث عن JSON مستقل خارج code fence كخطة بديلة
    لدعم النماذج التي لا تُغلّف JSON دائماً بـ code fence.
    """
    # المسار الأول: JSON داخل code fence (مفضّل)
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            parsed = json.loads(m.group(1))
            if "tool" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    # المسار الثاني: JSON خارج code fence (خطة بديلة)
    m2 = re.search(r"(\{[^{}]*\"tool\"[^{}]*\})", text, re.DOTALL)
    if m2:
        try:
            parsed = json.loads(m2.group(1))
            if "tool" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    return None


def execute_script(code: str) -> str:
    """تنفيذ سكريبت Python مباشرةً مع وراثة متغيرات البيئة الكاملة."""
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=60,
            env=os.environ.copy(),
        )
        output = result.stdout.strip()
        errors = result.stderr.strip()
        if result.returncode == 0:
            return f"✅ نتيجة التنفيذ:\n{output}" if output else "✅ تم التنفيذ بنجاح (لا مخرجات)."
        return f"❌ خطأ في التنفيذ (code={result.returncode}):\n{errors}\n{output}"
    except subprocess.TimeoutExpired:
        return "❌ انتهت مهلة التنفيذ (60 ثانية)."
    except Exception as e:
        return f"❌ خطأ غير متوقع أثناء تشغيل السكريبت: {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def execute_tool(action: dict, owner: str, repo: str) -> str:
    tool = action.get("tool", "")

    if tool == "push_file":
        path    = action.get("path", "output.txt")
        content = action.get("content", "")
        message = action.get("message", f"feat: push {path} via Selfe Agent")
        try:
            commit_url = push_file_to_github(owner, repo, path, content, message)
            return f"✅ تم رفع `{path}` بنجاح.\ncommit: {commit_url}"
        except Exception as e:
            return f"❌ فشل push_file: {e}"

    elif tool == "read_file":
        path = action.get("path", "")
        if not path:
            return "❌ يجب تحديد path."
        content = read_file_from_github(owner, repo, path)
        if content is None:
            return f"❌ لم يُعثر على `{path}`. تحقق من المسار أو استخدم list_files."
        if len(content) > 8000:
            content = content[:8000] + "\n...[truncated]"
        return f"📄 محتوى `{path}`:\n```\n{content}\n```"

    elif tool == "list_files":
        path = action.get("path", "")
        api_path = f"/repos/{owner}/{repo}/contents"
        if path and path != "/":
            api_path += f"/{path.strip('/')}"
        try:
            items = _github_request("GET", api_path)
            if not isinstance(items, list):
                return f"❌ لم يُعثر على المجلد `{path}`."
            lines = []
            for item in items:
                icon = "📁" if item.get("type") == "dir" else "📄"
                lines.append(f"{icon} {item['name']}")
            return f"📂 محتويات `{'/' if not path else path}`:\n" + "\n".join(lines)
        except Exception as e:
            return f"❌ خطأ list_files: {e}"

    elif tool == "search_files":
        query = action.get("query", "")
        if not query:
            return "❌ يجب تحديد query للبحث."
        try:
            items = _github_request("GET", f"/repos/{owner}/{repo}/git/trees/main?recursive=1")
            tree  = items.get("tree", [])
            matches = [
                f"{'📁' if i.get('type') == 'tree' else '📄'} {i['path']}"
                for i in tree
                if query.lower() in i.get("path", "").lower()
            ]
            if not matches:
                return f"🔍 لا نتائج لـ `{query}`."
            return f"🔍 نتائج البحث عن `{query}`:\n" + "\n".join(matches[:30])
        except Exception as e:
            return f"❌ خطأ search_files: {e}"

    elif tool == "run_script":
        code = action.get("code", "")
        if not code.strip():
            return "❌ يجب تحديد code للتنفيذ."

        # v6.4.0: تحقق من المتغيرات قبل التنفيذ
        service = detect_service_in_code(code)
        if service:
            ok, err_msg = check_required_env(service)
            if not ok:
                return err_msg

        print(f"[ReAct/run_script] تنفيذ سكريبت ({len(code)} حرف)...")
        raw_output = execute_script(code)

        # v6.4.0: التحقق من النتيجة
        return validate_execution_result(raw_output, action)

    elif tool == "send_email":
        # v6.5.0: أداة إرسال البريد الإلكتروني المباشرة
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        smtp_user = os.environ.get("SMTP_USER", "")
        smtp_pass = os.environ.get("SMTP_PASS", "")
        to        = action.get("to", "")
        subject   = action.get("subject", "رسالة من Selfe Agent")
        body      = action.get("body", "")
        if not smtp_user or not smtp_pass:
            return "❌ SMTP_USER أو SMTP_PASS غير موجودَين."
        if not to:
            return "❌ يجب تحديد عنوان المستلم في حقل 'to'."
        try:
            msg = MIMEMultipart("alternative")
            msg["From"]    = smtp_user
            msg["To"]      = to
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain", "utf-8"))
            with smtplib.SMTP("smtp.gmail.com", 587) as s:
                s.ehlo()
                s.starttls()
                s.login(smtp_user, smtp_pass)
                s.sendmail(smtp_user, to, msg.as_string())
            return f"✅ تم إرسال البريد إلى {to} بنجاح."
        except Exception as e:
            return f"❌ فشل إرسال البريد: {e}"

    elif tool == "answer":
        return action.get("text", "")

    else:
        return (
            f"⚠️ أداة غير معروفة: `{tool}`. "
            f"الأدوات المتاحة: read_file, push_file, list_files, search_files, run_script, send_email, answer"
        )


def react_loop(
    smart_client: "SmartAPIClient",
    model_name: str,
    messages: list,
    owner: str,
    repo: str,
    max_steps: int = 10,
) -> tuple:
    total_tokens  = 0
    tool_fail_count: Dict[str, int] = {}
    no_tool_count = 0

    for step in range(1, max_steps + 1):
        active_model = smart_client.active_model
        print(f"[ReAct] خطوة {step}/{max_steps}")

        try:
            resp = smart_client.chat_completions_create(
                step=step,
                model=active_model,
                messages=messages,
                temperature=0.2,
                max_tokens=4096,
            )
            raw = (resp.choices[0].message.content or "").strip()
            tokens = getattr(resp.usage, "total_tokens", 0)
            total_tokens += tokens
        except Exception as e:
            return f"⚠️ خطأ API لا يمكن التعافي منه (خطوة {step}): {e}", total_tokens

        print(f"[ReAct] رد النموذج:\n{raw[:300]}...")

        if not raw:
            no_tool_count += 1
            print(f"[ReAct] رد فارغ (المرة {no_tool_count}).")
            if no_tool_count >= 3:
                print("[ReAct] ثلاثة ردود فارغة/بدون أداة → إنهاء.")
                return "⚠️ لم يُرجع النموذج أي محتوى.", total_tokens
            messages.append({"role": "assistant", "content": raw or "[empty]"})
            messages.append({
                "role": "user",
                "content": (
                    "⚠️ ردك كان فارغاً. تذكّر أن ترد بأداة واحدة داخل ```json ... ```\n"
                    "إذا انتهيت من المهمة استخدم: ```json\n{\"tool\": \"answer\", \"text\": \"ردك\"}\n```"
                )
            })
            continue

        action = parse_tool_call(raw)

        if action is None:
            no_tool_count += 1
            print(f"[ReAct] لم تُعثر على أداة (المرة {no_tool_count}).")
            if no_tool_count >= 3:
                print("[ReAct] ثلاثة ردود بدون أداة → اعتبار الرد نهائياً.")
                return raw, total_tokens
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    "⛔ **خطأ: لم أتلقَّ أداة بالصيغة المطلوبة.**\n\n"
                    "**تذكير بمبدأ الاكتفاء الذاتي:** إذا لم تجد أداةً مُعرَّفة لمهمتك، "
                    "اكتب سكريبت Python بنفسك وشغّله عبر run_script.\n\n"
                    "**القاعدة الذهبية:** لا تصف ما ستفعله — افعله مباشرةً.\n"
                    "❌ خاطئ: \"سأقوم أولاً بقراءة الملف ثم...\"\n"
                    "✅ صحيح: ابدأ فوراً بـ JSON:\n"
                    "```json\n{\"tool\": \"read_file\", \"path\": \"agent_ci.py\"}\n```\n\n"
                    "- يجب أن يحتوي ردك على أداة واحدة داخل ```json ... ```\n"
                    "- إذا انتهيت من المهمة: ```json\n{\"tool\": \"answer\", \"text\": \"ردك\"}\n```\n"
                    "- إذا لم تنتهِ: تابع مع الأداة التالية المناسبة."
                )
            })
            continue

        no_tool_count = 0
        tool_name = action.get("tool", "")

        if tool_name == "answer":
            final = action.get("text", raw)
            print(f"[ReAct] ✅ اكتمال في الخطوة {step}.")
            return final, total_tokens

        observation = execute_tool(action, owner, repo)
        print(f"[ReAct] Observation: {observation[:200]}")

        if observation.startswith("❌"):
            tool_fail_count[tool_name] = tool_fail_count.get(tool_name, 0) + 1
            fail_count = tool_fail_count[tool_name]
            print(f"[ReAct] ⚠ أداة `{tool_name}` فشلت {fail_count}/3.")
            if fail_count >= 3:
                print(f"[ReAct] 🚫 `{tool_name}` فشلت 3 مرات — طلب مسار بديل.")
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Observation: {observation}\n\n"
                        f"🚫 فشلت أداة `{tool_name}` {fail_count} مرات متتالية. "
                        f"لا تُعيد استخدامها. "
                        f"تذكّر مبدأ الاكتفاء الذاتي: إذا فشلت الأداة المُعرَّفة، "
                        f"اكتب سكريبت Python بنفسك عبر run_script لتنفيذ المهمة مباشرةً، "
                        f"أو استخدم tool=answer مع شرح ما تعذّر إنجازه."
                    )
                })
                continue

        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    print(f"[ReAct] ⚠ وصلنا للحد الأقصى ({max_steps} خطوات). طلب رد نهائي...")
    messages.append({
        "role": "user",
        "content": (
            f"انتهت جميع الخطوات ({max_steps}/{max_steps}). "
            "لخّص ما أُنجز وما تعذّر إنجازه، ثم أجب نهائياً باستخدام:\n"
            "```json\n{\"tool\": \"answer\", \"text\": \"ردك النهائي\"}\n```"
        )
    })
    try:
        active_model = smart_client.active_model
        resp = smart_client.chat_completions_create(
            step=max_steps + 1,
            model=active_model,
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
        )
        raw    = (resp.choices[0].message.content or "").strip()
        total_tokens += getattr(resp.usage, "total_tokens", 0)
        action = parse_tool_call(raw)
        if action and action.get("tool") == "answer":
            return action.get("text", raw), total_tokens
        return raw, total_tokens
    except Exception as e:
        return f"⚠️ خطأ ReAct نهائي: {e}", total_tokens


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
                print(f"[Memory] issue #{self.issue_number} — {len(self._issue_data.get('turns',[]))} turn(s)")
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
            messages[0]["content"] += f"\n\n## سياق آخر {len(recent)} تفاعل:\n"
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
            print(f"[Memory] ⚠ {e}")
        try:
            existing = read_file_from_github(self.owner, self.repo, self.log_path) or ""
            entry    = json.dumps({"ts": self._now(), "issue": self.issue_number,
                                   "turn": len(turns), "model": model_name,
                                   "tokens": tokens, "success": success}, ensure_ascii=False)
            push_file_to_github(self.owner, self.repo, self.log_path,
                                existing.rstrip("\n") + "\n" + entry + "\n",
                                f"memory(log): issue-{self.issue_number}")
        except Exception as e:
            print(f"[Memory] log ⚠ {e}")

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ===================================================================
# SelfEvaluator
# ===================================================================

EVAL_SYSTEM_PROMPT = """أنت مُقيّم موضوعي. قيّم الرد من 0 إلى 10.
أجب بـ JSON فقط:
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
                model=self.smart_client.active_model,
                messages=[
                    {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                    {"role": "user",   "content": f"الطلب:\n{user_msg}\n\nالرد:\n{agent_reply}"},
                ],
                temperature=0.1, max_tokens=256,
            )
            raw = (resp.choices[0].message.content or "").strip()
            m   = re.search(r"\{.*\}", raw, re.DOTALL)
            return json.loads(m.group()) if m else {"score": 10, "issues": [], "improvements": [], "refined_prompt_addition": ""}
        except Exception as e:
            print(f"[SelfEval] ⚠ {e}")
            return {"score": 10, "issues": [], "improvements": [], "refined_prompt_addition": ""}

    def refine_system_prompt(self, original_prompt, eval_result):
        lines = []
        if eval_result.get("refined_prompt_addition", "").strip():
            lines.append(eval_result["refined_prompt_addition"])
        if eval_result.get("issues"):
            lines.append("تجنّب: " + "؛ ".join(eval_result["issues"]))
        if not lines:
            return original_prompt
        return original_prompt + "\n\n## توجيهات تلقائية:\n" + "\n".join(f"- {l}" for l in lines)

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
            print(f"[SelfEval] ⚠ log: {e}")

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
            print(f"[SelfEval] ⚠ stats: {e}")


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
    default = "أنت Selfe، وكيل ذكاء اصطناعي متخصص في تطوير البرمجيات."
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
        print(f"[ERROR] لم يُعثر على: {filepath}")
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
        print("[ERROR] لا توجد نماذج")
        sys.exit(1)
    return models


def detect_temperature(message):
    msg = message.lower()
    if any(k in msg for k in ["كود","code","دالة","function","اكتب","برمجة","debug","script","/push"]):
        return 0.2
    if any(k in msg for k in ["قصيدة","poem","اقتراح","فكرة","إبداع","تصميم","قصة"]):
        return 0.9
    if any(k in msg for k in ["ما هو","what is","اشرح","explain","فرق","difference"]):
        return 0.5
    return 0.7


def write_output(reply):
    gh_output = os.environ.get("GITHUB_OUTPUT", "")
    if gh_output:
        with open(gh_output, "a", encoding="utf-8") as fh:
            fh.write(f"reply<<EOF\n{reply}\nEOF\n")
    print("[Selfe Reply]")
    print(reply)


PUSH_SYSTEM_PROMPT = """أنت Selfe، وكيل برمجة.
اكتب الكود فقط داخل code fence:
```python
# الكود هنا
```"""


# ===================================================================
# main — v6.5.0
# ===================================================================

def main():
    print("\n[Selfe Agent CI v6.5.0] تشغيل...")

    msg = os.environ.get("USER_MESSAGE", "").strip()
    if not msg:
        write_output("⚠️ لم يتم استقبال أي رسالة.")
        sys.exit(0)

    gh_repo      = os.environ.get("GITHUB_REPOSITORY", "/")
    owner, repo  = gh_repo.split("/", 1)
    issue_number = os.environ.get("ISSUE_NUMBER", "0")

    models = load_models(MODELS_FILE)
    idx    = int(os.environ.get("MODEL_INDEX", "1")) - 1
    model  = models[idx] if idx < len(models) else models[0]
    print(f"[CI] النموذج: {model['name']} | {model['provider']} | Issue #{issue_number}")

    all_keys = {
        p: [os.environ.get(v, "").strip() for v in cfg["secret_vars"] if os.environ.get(v, "").strip()]
        for p, cfg in PROVIDER_CONFIG.items()
    }
    pkeys = all_keys.get(model["provider"], [])
    if not pkeys:
        write_output(f"⚠️ لا يوجد مفتاح API لـ: {model['provider']}")
        sys.exit(1)

    groq_keys     = all_keys.get("groq", [])
    groq_base_url = PROVIDER_CONFIG["groq"]["base_url"]
    fallback_keys      = groq_keys if model["provider"] == "gemini" and groq_keys else []
    fallback_base_url  = groq_base_url if fallback_keys else None
    fallback_model_name = GROQ_FALLBACK_MODEL if fallback_keys else None

    if fallback_keys:
        print(f"[CI] ✅ Groq Fallback جاهز ({GROQ_FALLBACK_MODEL})")
    else:
        print("[CI] ⚠ لا يوجد Groq Fallback (GROQ_API_KEY غير موجود أو المزود الأصلي هو Groq)")

    memory = MemoryManager(owner, repo, issue_number)

    try:
        from openai import OpenAI
    except ImportError:
        write_output("⚠️ مكتبة openai غير مثبَّتة.")
        sys.exit(1)

    cfg          = PROVIDER_CONFIG[model["provider"]]
    monitor      = ErrorMonitor(owner, repo, issue_number, model["name"])
    smart_client = SmartAPIClient(
        keys=pkeys,
        base_url=cfg["base_url"],
        model_name=model["name"],
        monitor=monitor,
        fallback_keys=fallback_keys,
        fallback_base_url=fallback_base_url,
        fallback_model=fallback_model_name,
    )
    evaluator = SelfEvaluator(owner, repo, smart_client, model["name"])

    # ── /push ─────────────────────────────────────────────────────
    is_push, push_instruction = detect_push_command(msg)
    if is_push:
        print(f"[CI] /push — {push_instruction}")
        filename   = extract_filename(push_instruction)
        max_push_retries = RETRY_STRATEGY["default"]["max_attempts"]
        for attempt in range(max_push_retries):
            try:
                resp = smart_client.chat_completions_create(
                    step=0,
                    model=smart_client.active_model,
                    messages=[
                        {"role": "system", "content": PUSH_SYSTEM_PROMPT},
                        {"role": "user",   "content": push_instruction},
                    ],
                    temperature=0.2, max_tokens=4096,
                )
                raw_reply  = (resp.choices[0].message.content or "").strip()
                clean_code = extract_code_from_reply(raw_reply)
                tokens     = getattr(resp.usage, "total_tokens", 0)
                commit_url = push_file_to_github(
                    owner, repo, filename, clean_code,
                    f"feat({filename}): generated by Selfe Agent via /push",
                )
                reply = (
                    f"✅ **تم دفع الملف بنجاح!**\n\n"
                    f"📄 **الملف:** `{filename}`\n"
                    f"🔗 **الـ commit:** {commit_url}\n\n"
                    f"```\n{clean_code[:1500]}\n```"
                )
                write_output(reply)
                memory.save_turn(msg, reply, model["name"], tokens, 0.2, True)
                monitor.flush_to_github()
                return
            except Exception as e:
                if attempt >= max_push_retries - 1:
                    monitor.flush_to_github()
                    write_output(f"⚠️ خطأ /push: {e}")
                    sys.exit(1)

    # ── ReAct Loop ─────────────────────────────────────────────────
    if is_complex_task(msg):
        print(f"[CI] ✨ تفعيل ReAct Loop")
        base_sp      = load_system_prompt(SYSTEM_PROMPT_FILE)
        data         = memory.load_issue_memory()
        current_turn = len(data.get("turns", [])) + 1

        messages = memory.build_messages(REACT_SYSTEM_PROMPT + "\n\n" + base_sp, msg)
        final_reply, total_tokens = react_loop(
            smart_client, smart_client.active_model, messages, owner, repo, max_steps=10
        )

        eval_result = evaluator.evaluate(msg, final_reply)
        score       = eval_result.get("score", 10)
        print(f"[SelfEval/ReAct] {score}/10")

        if score < EVAL_THRESHOLD:
            print(f"[SelfEval/ReAct] ⚠ score={score} < {EVAL_THRESHOLD} → إعادة محاولة")
            refined_sp = evaluator.refine_system_prompt(
                REACT_SYSTEM_PROMPT + "\n\n" + base_sp, eval_result
            )
            messages2 = memory.build_messages(refined_sp, msg)
            retry_reply, retry_tokens = react_loop(
                smart_client, smart_client.active_model, messages2, owner, repo, max_steps=10
            )
            eval_result2 = evaluator.evaluate(msg, retry_reply)
            score2       = eval_result2.get("score", 10)
            print(f"[SelfEval/ReAct] retry → {score2}/10")

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
            final_reply += f"\n\n---\n> ⚠️ *أفضل تقييم ذاتي: {score}/10*"

        write_output(final_reply)
        memory.save_turn(
            msg, final_reply, model["name"], total_tokens, 0.2,
            success=(score >= EVAL_THRESHOLD)
        )
        monitor.flush_to_github()
        print(f"[CI] ReAct اكتمل ✔  (score={score})")
        return

    # ── وضع عادي (Single-shot) + SelfEval + Memory ────────────────
    base_system_prompt    = load_system_prompt(SYSTEM_PROMPT_FILE)
    temp                  = detect_temperature(msg)
    current_system_prompt = base_system_prompt
    final_reply           = None
    best_score            = 0
    best_reply            = None
    final_tokens          = 0

    data         = memory.load_issue_memory()
    current_turn = len(data.get("turns", [])) + 1

    for attempt in range(MAX_SELF_EVAL_RETRIES + 1):
        messages = memory.build_messages(current_system_prompt, msg)
        try:
            resp = smart_client.chat_completions_create(
                step=0,
                model=smart_client.active_model,
                messages=messages,
                temperature=temp,
                max_tokens=4096,
            )
            raw_reply = (resp.choices[0].message.content or "").strip()
            tokens    = getattr(resp.usage, "total_tokens", 0)
            final_tokens += tokens
        except Exception as e:
            write_output(f"⚠️ خطأ API: {e}")
            monitor.flush_to_github()
            sys.exit(1)

        eval_result = evaluator.evaluate(msg, raw_reply)
        score       = eval_result.get("score", 10)
        print(f"[SelfEval] attempt={attempt} score={score}/10")

        if score > best_score:
            best_score = score
            best_reply = raw_reply

        evaluator.log_evaluation(issue_number, current_turn, msg, score, model["name"], attempt, improved=(attempt > 0))

        if score >= EVAL_THRESHOLD:
            break

        if attempt < MAX_SELF_EVAL_RETRIES:
            current_system_prompt = evaluator.refine_system_prompt(base_system_prompt, eval_result)

    final_reply = best_reply or raw_reply
    if best_score < EVAL_THRESHOLD:
        final_reply += f"\n\n---\n> ⚠️ *أفضل تقييم ذاتي: {best_score}/10*"

    evaluator.update_prompt_stats(best_score, improved=(best_score < EVAL_THRESHOLD))
    write_output(final_reply)
    memory.save_turn(msg, final_reply, model["name"], final_tokens, temp, success=(best_score >= EVAL_THRESHOLD))
    monitor.flush_to_github()
    print(f"[CI] اكتمل ✔  (best_score={best_score})")


if __name__ == "__main__":
    main()
