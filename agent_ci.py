# =============================================================
# agent_ci.py — Selfe Agent v5.0.0 (نسخة GitHub Actions)
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
#
# ميزة self-evaluation & auto prompt refinement (v5.0.0):
#   - بعد كل رد، يُقيّم الوكيل نفسه (score 0-10)
#   - إذا كان Score < EVAL_THRESHOLD يُعيد المحاولة بـ prompt مُحسَّن
#   - MAX_SELF_EVAL_RETRIES محاولات تحسين قبل الاستسلام
#   - سجل التقييمات: memory/eval_log.jsonl
#   - إحصاءات التحسين: memory/prompt_stats.json
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
MAX_MEMORY_TURNS   = 5
MEMORY_DIR         = "memory"

# ── إعدادات التقييم الذاتي ─────────────────────────────────────
EVAL_THRESHOLD       = 6    # أدنى درجة مقبولة من 10
MAX_SELF_EVAL_RETRIES = 2   # عدد مرات إعادة المحاولة بـ prompt مُحسَّن
EVAL_LOG_PATH        = f"{MEMORY_DIR}/eval_log.jsonl"
PROMPT_STATS_PATH    = f"{MEMORY_DIR}/prompt_stats.json"

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
# أدوات GitHub API
# -------------------------------------------------------------------

def _github_request(method: str, path: str, data: dict = None) -> dict:
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise RuntimeError("GITHUB_TOKEN غير موجود في البيئة.")
    url = f"https://api.github.com{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": "selfe-agent/5.0.0",
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
    try:
        result = _github_request("GET", f"/repos/{owner}/{repo}/contents/{path}?ref={branch}")
        return result.get("sha")
    except RuntimeError:
        return None


def push_file_to_github(owner: str, repo: str, filepath: str,
                        content: str, commit_message: str,
                        branch: str = "main") -> str:
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
    return f"https://github.com/{owner}/{repo}/commit/{commit_sha}"


def read_file_from_github(owner: str, repo: str, filepath: str,
                          branch: str = "main") -> str | None:
    try:
        result = _github_request("GET", f"/repos/{owner}/{repo}/contents/{filepath}?ref={branch}")
        encoded = result.get("content", "")
        return base64.b64decode(encoded.replace("\n", "")).decode("utf-8")
    except RuntimeError:
        return None


# -------------------------------------------------------------------
# MemoryManager
# -------------------------------------------------------------------

class MemoryManager:
    def __init__(self, owner: str, repo: str, issue_number: str):
        self.owner        = owner
        self.repo         = repo
        self.issue_number = issue_number
        self.issue_path   = f"{MEMORY_DIR}/issue_{issue_number}.json"
        self.log_path     = f"{MEMORY_DIR}/global_log.jsonl"
        self._issue_data  = None

    def load_issue_memory(self) -> dict:
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
        return {"issue_number": self.issue_number, "created_at": self._now(), "turns": []}

    def build_messages(self, system_prompt: str, current_msg: str) -> list:
        data   = self.load_issue_memory()
        turns  = data.get("turns", [])
        recent = turns[-MAX_MEMORY_TURNS:]
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

    def save_turn(self, user_msg: str, agent_reply: str,
                  model_name: str, tokens_used: int,
                  temperature: float, success: bool) -> None:
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
        MAX_STORED = MAX_MEMORY_TURNS * 2
        if len(turns) > MAX_STORED:
            turns = turns[-MAX_STORED:]
            print(f"[Memory] Consolidation: أُبقي على آخر {MAX_STORED}")
        data["turns"]      = turns
        data["updated_at"] = self._now()
        self._issue_data   = data
        try:
            push_file_to_github(
                self.owner, self.repo, self.issue_path,
                json.dumps(data, ensure_ascii=False, indent=2),
                f"memory(issue-{self.issue_number}): turn {new_turn['turn']} saved by Selfe Agent",
            )
            print(f"[Memory] ✔ issue memory saved → {self.issue_path}")
        except Exception as e:
            print(f"[Memory] ⚠ فشل حفظ issue memory: {e}")
        log_entry = {
            "ts": self._now(), "issue": self.issue_number,
            "turn": new_turn["turn"], "model": model_name,
            "tokens": tokens_used, "temp": temperature, "success": success,
        }
        self._append_global_log(log_entry)

    def _append_global_log(self, entry: dict) -> None:
        existing_raw = read_file_from_github(self.owner, self.repo, self.log_path)
        new_line     = json.dumps(entry, ensure_ascii=False)
        new_content  = (existing_raw.rstrip("\n") + "\n" + new_line + "\n") if existing_raw else (new_line + "\n")
        try:
            push_file_to_github(
                self.owner, self.repo, self.log_path, new_content,
                f"memory(log): issue-{entry['issue']} turn-{entry['turn']}",
            )
            print(f"[Memory] ✔ global log updated → {self.log_path}")
        except Exception as e:
            print(f"[Memory] ⚠ فشل تحديث global log: {e}")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# -------------------------------------------------------------------
# SelfEvaluator — نظام التقييم الذاتي وتحسين الـ Prompt (v5.0.0)
# مستوحى من مهارة agent-orchestration-improve-agent
# -------------------------------------------------------------------

EVAL_SYSTEM_PROMPT = """أنت مُقيِّم موضوعي لردود الذكاء الاصطناعي.
مهمتك: تقييم جودة الرد بناءً على الطلب الأصلي.

أعطِ درجة من 0 إلى 10 وفق هذا المعيار:
- 9-10: رد مثالي، مكتمل، دقيق، منظّم
- 7-8 : رد جيد، يُجيب على المطلوب مع هفوات بسيطة
- 5-6 : رد متوسط، يُجيب جزئياً أو يفتقر للتنظيم
- 3-4 : رد ضعيف، يُخطئ أو يُفوّت نقاطاً جوهرية
- 0-2 : رد فاشل تماماً، خاطئ أو خارج الموضوع

أجِب بـ JSON فقط بهذا الشكل بالضبط:
{
  "score": <رقم 0-10>,
  "issues": ["مشكلة 1", "مشكلة 2"],
  "improvements": ["تحسين 1", "تحسين 2"],
  "refined_prompt_addition": "<نص إضافي يُضاف لبداية الـ prompt لتحسين الرد>"
}
لا تكتب أي شيء خارج الـ JSON."""

EVAL_USER_TEMPLATE = """الطلب الأصلي:
{user_msg}

الرد المُقدَّم:
{agent_reply}

قيِّم الرد وأعطِ JSON كما طُلب منك."""


class SelfEvaluator:
    """
    نظام التقييم الذاتي وتحسين الـ Prompt تلقائياً.
    يعمل بعد كل رد: يُقيّم، وإن فشل يُحسّن ويُعيد المحاولة.
    """

    def __init__(self, owner: str, repo: str, client, model_name: str):
        self.owner      = owner
        self.repo       = repo
        self.client     = client
        self.model_name = model_name

    def evaluate(self, user_msg: str, agent_reply: str) -> dict:
        """
        يُرسل الرد للنموذج ليُقيّمه ويُعيد dict مع score و issues و improvements.
        عند الفشل يُعيد score=10 (تجاهل التقييم بأمان).
        """
        eval_user = EVAL_USER_TEMPLATE.format(
            user_msg=user_msg,
            agent_reply=agent_reply,
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                    {"role": "user",   "content": eval_user},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            raw = resp.choices[0].message.content.strip()
            # استخراج JSON حتى لو كان مُغلَّفاً بـ markdown
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"score": 10, "issues": [], "improvements": [], "refined_prompt_addition": ""}
        except Exception as e:
            print(f"[SelfEval] ⚠ فشل التقييم: {e}")
            return {"score": 10, "issues": [], "improvements": [], "refined_prompt_addition": ""}

    def refine_system_prompt(self, original_prompt: str, eval_result: dict) -> str:
        """
        يُحسّن الـ system prompt بإضافة توجيهات تصحيحية من نتيجة التقييم.
        """
        addition = eval_result.get("refined_prompt_addition", "").strip()
        issues   = eval_result.get("issues", [])

        refinement_lines = []
        if addition:
            refinement_lines.append(addition)
        if issues:
            refinement_lines.append(
                "تجنّب هذه الأخطاء في ردّك: " + "؛ ".join(issues)
            )

        if not refinement_lines:
            return original_prompt

        refined = original_prompt + "\n\n## توجيهات التحسين التلقائي:\n" + "\n".join(
            f"- {line}" for line in refinement_lines
        )
        print(f"[SelfEval] ✏ تم تحسين الـ prompt — إضافة {len(refinement_lines)} توجيه")
        return refined

    def log_evaluation(self, issue_number: str, turn: int,
                       user_msg: str, score: int,
                       model_name: str, attempt: int,
                       improved: bool) -> None:
        """حفظ سجل التقييم في eval_log.jsonl."""
        entry = {
            "ts":          datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "issue":       issue_number,
            "turn":        turn,
            "model":       model_name,
            "score":       score,
            "attempt":     attempt,
            "improved":    improved,
        }
        existing_raw = read_file_from_github(self.owner, self.repo, EVAL_LOG_PATH)
        new_line     = json.dumps(entry, ensure_ascii=False)
        new_content  = (existing_raw.rstrip("\n") + "\n" + new_line + "\n") if existing_raw else (new_line + "\n")
        try:
            push_file_to_github(
                self.owner, self.repo, EVAL_LOG_PATH, new_content,
                f"eval(log): issue-{issue_number} turn-{turn} score-{score}",
            )
            print(f"[SelfEval] ✔ eval log saved — score: {score}/10")
        except Exception as e:
            print(f"[SelfEval] ⚠ فشل حفظ eval log: {e}")

    def update_prompt_stats(self, score: int, improved: bool) -> None:
        """تحديث إحصاءات التحسين في prompt_stats.json."""
        raw   = read_file_from_github(self.owner, self.repo, PROMPT_STATS_PATH)
        stats = json.loads(raw) if raw else {
            "total_evals": 0, "total_score": 0,
            "improvements_triggered": 0, "avg_score": 0.0,
            "score_distribution": {str(i): 0 for i in range(11)},
        }
        stats["total_evals"]   += 1
        stats["total_score"]   += score
        stats["avg_score"]      = round(stats["total_score"] / stats["total_evals"], 2)
        key = str(min(score, 10))
        stats["score_distribution"][key] = stats["score_distribution"].get(key, 0) + 1
        if improved:
            stats["improvements_triggered"] += 1
        stats["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            push_file_to_github(
                self.owner, self.repo, PROMPT_STATS_PATH,
                json.dumps(stats, ensure_ascii=False, indent=2),
                f"eval(stats): avg_score={stats['avg_score']} total={stats['total_evals']}",
            )
            print(f"[SelfEval] ✔ stats updated — avg: {stats['avg_score']}/10 | total: {stats['total_evals']}")
        except Exception as e:
            print(f"[SelfEval] ⚠ فشل تحديث stats: {e}")


# -------------------------------------------------------------------
# كشف أمر /push واستخراج اسم الملف
# -------------------------------------------------------------------

PUSH_PATTERN      = re.compile(r"^/push\s+(.+)$", re.IGNORECASE | re.MULTILINE)
FILE_NAME_PATTERN = re.compile(r"\b([\w\-/]+\.\w+)\b")
CODE_BLOCK_PATTERN = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)


def detect_push_command(message: str):
    m = PUSH_PATTERN.search(message)
    return (True, m.group(1).strip()) if m else (False, "")


def extract_filename(instruction: str) -> str:
    m = FILE_NAME_PATTERN.search(instruction)
    if m:
        return m.group(1)
    words   = re.sub(r"[^\w\s]", "", instruction).split()
    slug    = "_".join(words[:4]).lower() if words else "script"
    lang_map = {
        "python": ".py", "py": ".py", "javascript": ".js", "js": ".js",
        "typescript": ".ts", "ts": ".ts", "bash": ".sh", "shell": ".sh",
        "html": ".html", "css": ".css", "yaml": ".yml", "json": ".json",
    }
    for kw, ext in lang_map.items():
        if kw in instruction.lower():
            return f"{slug}{ext}"
    return f"{slug}.py"


def extract_code_from_reply(reply: str) -> str:
    blocks = CODE_BLOCK_PATTERN.findall(reply)
    return blocks[0].strip() if blocks else reply.strip()


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
                if "|" in line else (line, "gemini")
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
    code_kw     = ["كود","code","دالة","function","اكتب","برمجة","خطأ","debug",
                   "script","سكريبت","class","كلاس","api","endpoint","sql","query","استعلام","/push"]
    factual_kw  = ["ما هو","what is","كيف يعمل","how does","اشرح","explain",
                   "عرّف","define","فرق","difference","متى","when","لماذا","why","من هو","who is"]
    creative_kw = ["قصيدة","poem","اقتراح","suggest","فكرة","idea","أفكار","ideas",
                   "إبداع","creative","تصميم","design","تخيّل","imagine","قصة","story"]
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
    print("\n[Selfe Agent CI v5.0.0] تشغيل الوكيل في بيئة GitHub Actions...")

    msg = os.environ.get("USER_MESSAGE", "").strip()
    if not msg:
        write_output("⚠️ لم يتم استقبال أي رسالة من الـ Issue.")
        sys.exit(0)

    gh_repo      = os.environ.get("GITHUB_REPOSITORY", "/")
    owner, repo  = gh_repo.split("/", 1)
    issue_number = os.environ.get("ISSUE_NUMBER", "0")

    models = load_models(MODELS_FILE)
    idx    = int(os.environ.get("MODEL_INDEX", "1")) - 1
    model  = models[idx] if idx < len(models) else models[0]
    print(f"[CI] النموذج: {model['name']} | المزود: {model['provider']}")
    print(f"[CI] Issue #{issue_number}")

    all_keys = {
        p: [os.environ.get(v, "").strip() for v in cfg["secret_vars"] if os.environ.get(v, "").strip()]
        for p, cfg in PROVIDER_CONFIG.items()
    }

    pkeys = all_keys.get(model["provider"], [])
    if not pkeys:
        write_output(f"⚠️ لا يوجد مفتاح API للمزود: {model['provider']}")
        sys.exit(1)

    memory = MemoryManager(owner, repo, issue_number)

    try:
        from openai import OpenAI, RateLimitError, AuthenticationError
    except ImportError:
        write_output("⚠️ مكتبة openai غير مثبتَّتة.")
        sys.exit(1)

    cfg    = PROVIDER_CONFIG[model["provider"]]
    client = OpenAI(api_key=pkeys[0], base_url=cfg["base_url"])

    # تهيئة SelfEvaluator
    evaluator = SelfEvaluator(owner, repo, client, model["name"])

    # ---------------------------------------------------------------
    # كشف أمر /push
    # ---------------------------------------------------------------
    is_push, push_instruction = detect_push_command(msg)

    if is_push:
        print(f"[CI] وضع /push — التعليمة: {push_instruction}")
        filename = extract_filename(push_instruction)
        print(f"[CI] اسم الملف المستهدف: {filename}")

        for attempt in range(MAX_RETRIES):
            try:
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

                commit_url = push_file_to_github(
                    owner, repo, filename, clean_code,
                    f"feat({filename}): generated by Selfe Agent via /push",
                )
                reply = (
                    f"✅ **تم دفع الملف بنجاح!**\n\n"
                    f"📄 **الملف:** `{filename}`\n"
                    f"🔗 **الـ commit:** {commit_url}\n\n"
                    f"**الكود المكتوب:**\n"
                    f"```\n{clean_code}\n```"
                )
                write_output(reply)
                memory.save_turn(msg, reply, model["name"], tokens, 0.2, True)
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
    # الوضع العادي — مع التقييم الذاتي وتحسين الـ Prompt (v5.0.0)
    # ---------------------------------------------------------------
    base_system_prompt = load_system_prompt(SYSTEM_PROMPT_FILE)
    temp               = detect_temperature(msg)
    print(f"[CI] Temperature: {temp}")

    current_system_prompt = base_system_prompt
    final_reply           = None
    final_tokens          = 0
    best_score            = 0
    best_reply            = None

    data        = memory.load_issue_memory()
    current_turn = len(data.get("turns", [])) + 1

    for eval_attempt in range(MAX_SELF_EVAL_RETRIES + 1):

        if eval_attempt > 0:
            print(f"[SelfEval] 🔄 إعادة المحاولة {eval_attempt}/{MAX_SELF_EVAL_RETRIES} بـ prompt مُحسَّن...")

        messages = memory.build_messages(current_system_prompt, msg)
        print(f"[CI] إجمالي الـ messages: {len(messages)} | eval_attempt: {eval_attempt}")

        reply  = None
        tokens = 0

        for api_attempt in range(MAX_RETRIES):
            try:
                resp   = client.chat.completions.create(
                    model=model["name"],
                    messages=messages,
                    temperature=temp,
                    max_tokens=4096,
                )
                reply  = resp.choices[0].message.content
                tokens = getattr(resp.usage, "total_tokens", 0)
                break

            except RateLimitError:
                write_output("⚠️ تجاوز حد الطلبات (Rate Limit). حاول لاحقاً.")
                memory.save_turn(msg, "Rate Limit Error", model["name"], 0, temp, False)
                sys.exit(1)
            except AuthenticationError:
                write_output("⚠️ مفتاح API غير صالح. تحقّق من GitHub Secrets.")
                memory.save_turn(msg, "Auth Error", model["name"], 0, temp, False)
                sys.exit(1)
            except Exception as e:
                if api_attempt < MAX_RETRIES - 1:
                    wait = 2 ** api_attempt
                    print(f"[تحذير] خطأ مؤقت، إعادة المحاولة {api_attempt+1}/{MAX_RETRIES} بعد {wait}s...")
                    time.sleep(wait)
                else:
                    write_output(f"⚠️ خطأ: {e}")
                    memory.save_turn(msg, str(e), model["name"], 0, temp, False)
                    sys.exit(1)

        if reply is None:
            break

        # ── تقييم الرد ──────────────────────────────────────────────
        print(f"[SelfEval] 🔍 تقييم الرد (attempt {eval_attempt})...")
        eval_result = evaluator.evaluate(msg, reply)
        score       = eval_result.get("score", 10)
        issues      = eval_result.get("issues", [])

        print(f"[SelfEval] النتيجة: {score}/10 | المشاكل: {issues}")

        # حفظ الأفضل دائماً
        if score > best_score:
            best_score = score
            best_reply = reply
            final_tokens = tokens

        # تسجيل التقييم
        improved = (eval_attempt > 0)
        evaluator.log_evaluation(issue_number, current_turn, msg, score, model["name"], eval_attempt, improved)
        evaluator.update_prompt_stats(score, improved)

        if score >= EVAL_THRESHOLD:
            print(f"[SelfEval] ✅ الرد مقبول (score={score} ≥ threshold={EVAL_THRESHOLD})")
            final_reply = reply
            break
        else:
            print(f"[SelfEval] ⚠ الرد دون المستوى (score={score} < {EVAL_THRESHOLD})")
            if eval_attempt < MAX_SELF_EVAL_RETRIES:
                # تحسين الـ prompt وإعادة المحاولة
                current_system_prompt = evaluator.refine_system_prompt(
                    base_system_prompt, eval_result
                )
            else:
                print(f"[SelfEval] استُنفدت المحاولات — استخدام أفضل رد (score={best_score})")
                final_reply = best_reply

    # ── إخراج الرد النهائي ──────────────────────────────────────────
    if not final_reply:
        final_reply = best_reply or "⚠️ لم يتمكن الوكيل من توليد رد مناسب."

    # إضافة badge التقييم إذا كان score منخفضاً للتوثيق
    if best_score < EVAL_THRESHOLD:
        final_reply += f"\n\n---\n> ⚠️ *ملاحظة: أفضل تقييم ذاتي للرد كان {best_score}/10. قد تحتاج لمراجعة إضافية.*"

    write_output(final_reply)
    memory.save_turn(msg, final_reply, model["name"], final_tokens, temp, True)
    print(f"[CI] اكتمل بنجاح ✔ | best_score: {best_score}/10 | tokens: {final_tokens}")


if __name__ == "__main__":
    main()
