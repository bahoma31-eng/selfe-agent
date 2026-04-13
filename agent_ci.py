# =============================================================
# agent_ci.py — Selfe Agent v4.0.0 (نسخة GitHub Actions)
# تعمل بدون تفاعل (non-interactive)
# تقرأ USER_MESSAGE و MODEL_INDEX و ISSUE_NUMBER من environment variables
# تكتب الرد في GITHUB_OUTPUT ليُنشر كـ comment على الـ Issue
#
# ميزة /push:
#   اكتب في الـ Issue أو Comment:
#   /push اكتب سكريبت Python يحسب الفيبوناتشي واحفظه في fibonacci.py
#   سيقوم الوكيل بكتابة الكود ودفعه مباشرةً إلى main
#
# ميزة memory-systems (v4.0.0):
#   - ذاكرة قصيرة المدى: memory/issue_{N}.json (آخر MAX_MEMORY_TURNS محادثات)
#   - logging طويل المدى: memory/global_log.jsonl (كل طلب)
#   - الذاكرة محفوظة في المستودع عبر GitHub Contents API
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

MODELS_FILE        = "models.txt"
SYSTEM_PROMPT_FILE = "system_prompt.txt"
MAX_RETRIES        = 3
MAX_MEMORY_TURNS   = 5   # أقصى عدد من التفاعلات السابقة تُحقن في الـ context
MEMORY_DIR         = "memory"

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

# -------------------------------------------------------------------
# أدوات GitHub API  (مستوحاة من مهارة github-automation)
# -------------------------------------------------------------------

def _github_request(method: str, path: str, data: dict = None) -> dict:
    """طلب مباشر لـ GitHub Contents API باستخدام GITHUB_TOKEN."""
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise RuntimeError("GITHUB_TOKEN غير موجود في البيئة.")

    url = f"https://api.github.com{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": "selfe-agent/4.0.0",
    }
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        raise RuntimeError(f"GitHub API {e.code}: {err_body}")


def get_file_sha(owner: str, repo: str, path: str, branch: str = "main") -> str | None:
    """إرجاع SHA الملف الحالي إن وُجد، وإلا None."""
    try:
        result = _github_request("GET", f"/repos/{owner}/{repo}/contents/{path}?ref={branch}")
        return result.get("sha")
    except RuntimeError:
        return None


def push_file_to_github(owner: str, repo: str, filepath: str,
                        content: str, commit_message: str,
                        branch: str = "main") -> str:
    """
    دفع ملف إلى المستودع عبر GitHub Contents API.
    يُعيد رابط الـ commit.
    مستوحى من مهارة github-automation
    """
    existing_sha = get_file_sha(owner, repo, filepath, branch)
    payload = {
        "message": commit_message,
        "content": base64.b64encode(content.encode()).decode(),
        "branch": branch,
    }
    if existing_sha:
        payload["sha"] = existing_sha

    result = _github_request("PUT", f"/repos/{owner}/{repo}/contents/{filepath}", payload)
    commit_sha = result.get("commit", {}).get("sha", "")
    commit_url = f"https://github.com/{owner}/{repo}/commit/{commit_sha}"
    return commit_url


def read_file_from_github(owner: str, repo: str, filepath: str,
                          branch: str = "main") -> str | None:
    """قراءة محتوى ملف من المستودع. يُعيد None إذا لم يُوجد."""
    try:
        result = _github_request("GET", f"/repos/{owner}/{repo}/contents/{filepath}?ref={branch}")
        encoded = result.get("content", "")
        return base64.b64decode(encoded.replace("\n", "")).decode("utf-8")
    except RuntimeError:
        return None


# -------------------------------------------------------------------
# MemoryManager — مستوحى من مهارة memory-systems (Pattern 1: File-System)
# -------------------------------------------------------------------

class MemoryManager:
    """
    إدارة ذاكرة الوكيل بطبقتين حسب مهارة memory-systems:
      - Short-Term : memory/issue_{N}.json   — آخر MAX_MEMORY_TURNS turns لكل issue
      - Long-Term  : memory/global_log.jsonl — سجل شامل لكل الطلبات (logging)
    الذاكرة محفوظة في المستودع عبر GitHub API (بيئة Actions نظيفة عند كل run).
    """

    def __init__(self, owner: str, repo: str, issue_number: str):
        self.owner        = owner
        self.repo         = repo
        self.issue_number = issue_number
        self.issue_path   = f"{MEMORY_DIR}/issue_{issue_number}.json"
        self.log_path     = f"{MEMORY_DIR}/global_log.jsonl"
        self._issue_data  = None   # cache محلي

    # ── قراءة الذاكرة ──────────────────────────────────────────────

    def load_issue_memory(self) -> dict:
        """تحميل ذاكرة الـ issue من المستودع."""
        if self._issue_data is not None:
            return self._issue_data

        raw = read_file_from_github(self.owner, self.repo, self.issue_path)
        if raw:
            try:
                self._issue_data = json.loads(raw)
                turns_count = len(self._issue_data.get("turns", []))
                print(f"[Memory] وُجدت ذاكرة لـ issue #{self.issue_number} — {turns_count} turn(s)")
            except json.JSONDecodeError:
                self._issue_data = self._empty_issue()
        else:
            print(f"[Memory] لا توجد ذاكرة سابقة لـ issue #{self.issue_number} — بدء جديد")
            self._issue_data = self._empty_issue()

        return self._issue_data

    def _empty_issue(self) -> dict:
        return {
            "issue_number": self.issue_number,
            "created_at":   self._now(),
            "turns":        []
        }

    # ── بناء messages مع الذاكرة ───────────────────────────────────

    def build_messages(self, system_prompt: str, current_msg: str) -> list:
        """
        بناء قائمة الـ messages للـ LLM مع حقن الذاكرة.
        Pattern: Memory-Aware Prompting (conversation-memory skill)
        """
        data   = self.load_issue_memory()
        turns  = data.get("turns", [])
        recent = turns[-MAX_MEMORY_TURNS:]   # آخر N turns فقط

        messages = [{"role": "system", "content": system_prompt}]

        if recent:
            memory_note = (
                f"\n\n## سياق المحادثة السابقة (آخر {len(recent)} تفاعل)\n"
                "استخدم هذا السياق لفهم الطلب الحالي:\n"
            )
            messages[0]["content"] += memory_note

        for turn in recent:
            messages.append({"role": "user",      "content": turn["user"]})
            messages.append({"role": "assistant",  "content": turn["agent"]})

        messages.append({"role": "user", "content": current_msg})
        return messages

    # ── حفظ التفاعل بعد الرد ───────────────────────────────────────

    def save_turn(self, user_msg: str, agent_reply: str,
                  model_name: str, tokens_used: int,
                  temperature: float, success: bool) -> None:
        """
        حفظ التفاعل الجديد في:
          1. memory/issue_{N}.json  (Short-Term)
          2. memory/global_log.jsonl (Long-Term logging)
        ثم دفعهما للمستودع.
        """
        data  = self.load_issue_memory()
        turns = data.get("turns", [])

        new_turn = {
            "turn":        len(turns) + 1,
            "timestamp":   self._now(),
            "user":        user_msg,
            "agent":       agent_reply,
            "model":       model_name,
            "tokens":      tokens_used,
            "temperature": temperature,
            "success":     success,
        }
        turns.append(new_turn)

        # Consolidation: احتفظ بآخر MAX_MEMORY_TURNS * 2 فقط لمنع النمو اللانهائي
        MAX_STORED = MAX_MEMORY_TURNS * 2
        if len(turns) > MAX_STORED:
            turns = turns[-MAX_STORED:]
            print(f"[Memory] Consolidation: تم أرشفة التفاعلات القديمة، أُبقي على آخر {MAX_STORED}")

        data["turns"]      = turns
        data["updated_at"] = self._now()
        self._issue_data   = data

        gh_repo = f"{self.owner}/{self.repo}"
        try:
            push_file_to_github(
                self.owner, self.repo,
                self.issue_path,
                json.dumps(data, ensure_ascii=False, indent=2),
                f"memory(issue-{self.issue_number}): turn {new_turn['turn']} saved by Selfe Agent",
            )
            print(f"[Memory] ✔ issue memory saved → {self.issue_path}")
        except Exception as e:
            print(f"[Memory] ⚠ فشل حفظ issue memory: {e}")

        # Long-Term logging (JSONL — سطر واحد لكل طلب)
        log_entry = {
            "ts":       self._now(),
            "issue":    self.issue_number,
            "turn":     new_turn["turn"],
            "model":    model_name,
            "tokens":   tokens_used,
            "temp":     temperature,
            "success":  success,
        }
        self._append_global_log(log_entry)

    def _append_global_log(self, entry: dict) -> None:
        """إضافة سطر JSONL إلى global_log.jsonl."""
        existing_raw = read_file_from_github(self.owner, self.repo, self.log_path)
        new_line     = json.dumps(entry, ensure_ascii=False)
        if existing_raw:
            new_content = existing_raw.rstrip("\n") + "\n" + new_line + "\n"
        else:
            new_content = new_line + "\n"
        try:
            push_file_to_github(
                self.owner, self.repo,
                self.log_path,
                new_content,
                f"memory(log): issue-{entry['issue']} turn-{entry['turn']}",
            )
            print(f"[Memory] ✔ global log updated → {self.log_path}")
        except Exception as e:
            print(f"[Memory] ⚠ فشل تحديث global log: {e}")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# -------------------------------------------------------------------
# كشف أمر /push واستخراج اسم الملف
# -------------------------------------------------------------------

PUSH_PATTERN = re.compile(
    r"^/push\s+(.+)$",
    re.IGNORECASE | re.MULTILINE
)

FILE_NAME_PATTERN = re.compile(
    r"\b([\w\-/]+\.\w+)\b"
)

CODE_BLOCK_PATTERN = re.compile(
    r"```(?:\w+)?\n(.*?)```",
    re.DOTALL
)

def detect_push_command(message: str):
    m = PUSH_PATTERN.search(message)
    if m:
        return True, m.group(1).strip()
    return False, ""


def extract_filename(instruction: str) -> str:
    m = FILE_NAME_PATTERN.search(instruction)
    if m:
        return m.group(1)
    words = re.sub(r"[^\w\s]", "", instruction).split()
    slug  = "_".join(words[:4]).lower() if words else "script"
    lang_map = {
        "python": ".py", "py": ".py",
        "javascript": ".js", "js": ".js",
        "typescript": ".ts", "ts": ".ts",
        "bash": ".sh", "shell": ".sh",
        "html": ".html", "css": ".css",
        "yaml": ".yml", "json": ".json",
    }
    for kw, ext in lang_map.items():
        if kw in instruction.lower():
            return f"{slug}{ext}"
    return f"{slug}.py"


def extract_code_from_reply(reply: str) -> str:
    blocks = CODE_BLOCK_PATTERN.findall(reply)
    if blocks:
        return blocks[0].strip()
    return reply.strip()


# -------------------------------------------------------------------
# دوال مشتركة
# -------------------------------------------------------------------

def load_system_prompt(filepath: str) -> str:
    default = "أنت Selfe، وكيل ذكاء اصطناعي متخصص في تطوير البرمجيات."
    if not os.path.exists(filepath):
        return default
    lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip().startswith("#"):
                lines.append(line.rstrip())
    return "\n".join(lines).strip() or default


def load_models(filepath: str) -> list:
    if not os.path.exists(filepath):
        print(f"[ERROR] لم يُعثر على: {filepath}")
        sys.exit(1)
    models = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, provider = (
                [p.strip() for p in line.split("|", 1)]
                if "|" in line
                else (line, "gemini")
            )
            provider = provider.lower()
            if provider in PROVIDER_CONFIG:
                models.append({"name": name, "provider": provider})
    if not models:
        print("[ERROR] لا توجد نماذج في models.txt")
        sys.exit(1)
    return models


def detect_temperature(message: str) -> float:
    msg = message.lower()
    code_kw     = ["كود", "code", "دالة", "function", "اكتب", "برمجة",
                   "خطأ", "debug", "script", "سكريبت", "class", "كلاس",
                   "api", "endpoint", "sql", "query", "استعلام", "/push"]
    factual_kw  = ["ما هو", "what is", "كيف يعمل", "how does", "اشرح",
                   "explain", "عرّف", "define", "فرق", "difference",
                   "متى", "when", "لماذا", "why", "من هو", "who is"]
    creative_kw = ["قصيدة", "poem", "اقتراح", "suggest", "فكرة", "idea",
                   "أفكار", "ideas", "إبداع", "creative", "تصميم", "design",
                   "تخيّل", "imagine", "قصة", "story"]
    if any(k in msg for k in code_kw):     return 0.2
    if any(k in msg for k in creative_kw): return 0.9
    if any(k in msg for k in factual_kw):  return 0.5
    return 0.7


def write_output(reply: str) -> None:
    gh_output = os.environ.get("GITHUB_OUTPUT", "")
    if gh_output:
        with open(gh_output, "a", encoding="utf-8") as fh:
            fh.write(f"reply<<EOF\n{reply}\nEOF\n")
    print("[Selfe Reply]")
    print(reply)


# -------------------------------------------------------------------
# PUSH SYSTEM PROMPT
# -------------------------------------------------------------------

PUSH_SYSTEM_PROMPT = """أنت Selfe، وكيل برمجة متخصص.
مهمتك: كتابة الكود المطلوب فقط، بدون أي شرح أو تعليق خارج الكود.
أخرج الكود داخل code fence واحدة فقط. مثال:
```python
# الكود هنا
```
لا تكتب أي نص قبل أو بعد الـ code fence."""


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------

def main():
    print("\n[Selfe Agent CI v4.0.0] تشغيل الوكيل في بيئة GitHub Actions...")

    msg = os.environ.get("USER_MESSAGE", "").strip()
    if not msg:
        write_output("⚠️ لم يتم استقبال أي رسالة من الـ Issue.")
        sys.exit(0)

    # استخراج بيانات المستودع والـ issue
    gh_repo      = os.environ.get("GITHUB_REPOSITORY", "/")
    owner, repo  = gh_repo.split("/", 1)
    issue_number = os.environ.get("ISSUE_NUMBER", "0")

    models = load_models(MODELS_FILE)
    idx    = int(os.environ.get("MODEL_INDEX", "1")) - 1
    model  = models[idx] if idx < len(models) else models[0]
    print(f"[CI] النموذج: {model['name']} | المزود: {model['provider']}")
    print(f"[CI] Issue #{issue_number}")

    all_keys = {}
    for p, cfg in PROVIDER_CONFIG.items():
        all_keys[p] = [
            os.environ.get(v, "").strip()
            for v in cfg["secret_vars"]
            if os.environ.get(v, "").strip()
        ]

    pkeys = all_keys.get(model["provider"], [])
    if not pkeys:
        write_output(f"⚠️ لا يوجد مفتاح API للمزود: {model['provider']}")
        sys.exit(1)

    # تهيئة MemoryManager
    memory = MemoryManager(owner, repo, issue_number)

    # ---------------------------------------------------------------
    # كشف أمر /push
    # ---------------------------------------------------------------
    is_push, push_instruction = detect_push_command(msg)

    if is_push:
        print(f"[CI] وضع /push — التعليمة: {push_instruction}")
        filename = extract_filename(push_instruction)
        print(f"[CI] اسم الملف المستهدف: {filename}")

        try:
            from openai import OpenAI, RateLimitError, AuthenticationError
        except ImportError:
            write_output("⚠️ مكتبة openai غير مثبتَّتة.")
            sys.exit(1)

        cfg = PROVIDER_CONFIG[model["provider"]]

        for attempt in range(MAX_RETRIES):
            try:
                client = OpenAI(api_key=pkeys[0], base_url=cfg["base_url"])
                resp = client.chat.completions.create(
                    model=model["name"],
                    messages=[
                        {"role": "system", "content": PUSH_SYSTEM_PROMPT},
                        {"role": "user",   "content": push_instruction},
                    ],
                    temperature=0.2,
                    max_tokens=4096,
                )
                raw_reply  = resp.choices[0].message.content
                clean_code = extract_code_from_reply(raw_reply)
                tokens     = getattr(resp.usage, "total_tokens", 0)

                commit_msg = f"feat({filename}): generated by Selfe Agent via /push"
                print(f"[CI] دفع الملف إلى {owner}/{repo}/main/{filename} ...")
                commit_url = push_file_to_github(owner, repo, filename,
                                                  clean_code, commit_msg)

                reply = (
                    f"✅ **تم دفع الملف بنجاح!**\n\n"
                    f"📄 **الملف:** `{filename}`\n"
                    f"🔗 **الـ commit:** {commit_url}\n\n"
                    f"**الكود المكتوب:**\n"
                    f"```\n{clean_code}\n```"
                )
                write_output(reply)

                # حفظ في الذاكرة
                memory.save_turn(
                    user_msg=msg, agent_reply=reply,
                    model_name=model["name"], tokens_used=tokens,
                    temperature=0.2, success=True,
                )
                print("[CI] /push اكتمل بنجاح ✔")
                return

            except RateLimitError:
                write_output("⚠️ تجاوز حد الطلبات (Rate Limit). حاول لاحقاً.")
                memory.save_turn(msg, "Rate Limit Error", model["name"], 0, 0.2, False)
                sys.exit(1)
            except AuthenticationError:
                write_output("⚠️ مفتاح API غير صالح. تحقّق من GitHub Secrets.")
                memory.save_turn(msg, "Auth Error", model["name"], 0, 0.2, False)
                sys.exit(1)
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    print(f"[تحذير] خطأ، إعادة المحاولة {attempt+1}/{MAX_RETRIES} بعد {wait}s...")
                    time.sleep(wait)
                else:
                    write_output(f"⚠️ خطأ أثناء /push: {e}")
                    memory.save_turn(msg, str(e), model["name"], 0, 0.2, False)
                    sys.exit(1)

    # ---------------------------------------------------------------
    # الوضع العادي — رد نصي على الـ Issue مع الذاكرة
    # ---------------------------------------------------------------
    system_prompt = load_system_prompt(SYSTEM_PROMPT_FILE)
    temp          = detect_temperature(msg)
    print(f"[CI] Temperature: {temp}")

    try:
        from openai import OpenAI, RateLimitError, AuthenticationError
    except ImportError:
        write_output("⚠️ مكتبة openai غير مثبتَّتة.")
        sys.exit(1)

    cfg = PROVIDER_CONFIG[model["provider"]]

    # بناء messages مع حقن الذاكرة (memory-systems: Memory-Aware Prompting)
    messages = memory.build_messages(system_prompt, msg)
    print(f"[CI] إجمالي الـ messages المُرسلة: {len(messages)} (شامل الذاكرة)")

    for attempt in range(MAX_RETRIES):
        try:
            client = OpenAI(api_key=pkeys[0], base_url=cfg["base_url"])
            resp   = client.chat.completions.create(
                model=model["name"],
                messages=messages,
                temperature=temp,
                max_tokens=4096,
            )
            reply  = resp.choices[0].message.content
            tokens = getattr(resp.usage, "total_tokens", 0)

            write_output(reply)

            # حفظ في الذاكرة بعد الرد
            memory.save_turn(
                user_msg=msg, agent_reply=reply,
                model_name=model["name"], tokens_used=tokens,
                temperature=temp, success=True,
            )
            print(f"[CI] اكتمل بنجاح ✔ | tokens: {tokens}")
            return

        except RateLimitError:
            write_output("⚠️ تجاوز حد الطلبات (Rate Limit). حاول لاحقاً.")
            memory.save_turn(msg, "Rate Limit Error", model["name"], 0, temp, False)
            sys.exit(1)
        except AuthenticationError:
            write_output("⚠️ مفتاح API غير صالح. تحقّق من GitHub Secrets.")
            memory.save_turn(msg, "Auth Error", model["name"], 0, temp, False)
            sys.exit(1)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"[تحذير] خطأ مؤقت، إعادة المحاولة {attempt+1}/{MAX_RETRIES} بعد {wait}s...")
                time.sleep(wait)
            else:
                write_output(f"⚠️ خطأ: {e}")
                memory.save_turn(msg, str(e), model["name"], 0, temp, False)
                sys.exit(1)


if __name__ == "__main__":
    main()
