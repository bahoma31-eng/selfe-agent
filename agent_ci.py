# =============================================================
# agent_ci.py — Selfe Agent v5.1.0 (نسخة GitHub Actions)
# تعمل بدون تفاعل (non-interactive)
# جديد v5.1.0: ToolRegistry مدمج
# =============================================================

import os
import sys
import time
import re
import json
import base64
import inspect
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
# ToolRegistry (v5.1.0)
# -------------------------------------------------------------------

ToolType = TypeVar('ToolType', bound=Callable[..., Any])

class ToolRegistry:
    """سجل الأدوات المتاحة للوكيل."""

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


# -------------------------------------------------------------------
# أدوات GitHub API
# -------------------------------------------------------------------

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
        "User-Agent": "selfe-agent/5.1.0",
    }
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
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


# -------------------------------------------------------------------
# MemoryManager
# -------------------------------------------------------------------

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
            messages.append({"role": "user",      "content": t["user"]})
            messages.append({"role": "assistant",  "content": t["agent"]})
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


# -------------------------------------------------------------------
# SelfEvaluator
# -------------------------------------------------------------------

EVAL_SYSTEM_PROMPT = """أنت مُقيّم موضوعي. قيّم الرد من 0 إلى 10.
أجب بـ JSON فقط:
{"score":<0-10>,"issues":[],"improvements":[],"refined_prompt_addition":""}"""


class SelfEvaluator:
    def __init__(self, owner, repo, client, model_name):
        self.owner      = owner
        self.repo       = repo
        self.client     = client
        self.model_name = model_name

    def evaluate(self, user_msg, agent_reply):
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                    {"role": "user",   "content": f"الطلب:\n{user_msg}\n\nالرد:\n{agent_reply}"},
                ],
                temperature=0.1, max_tokens=256,
            )
            raw = resp.choices[0].message.content.strip()
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
        entry   = json.dumps({"ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
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


# -------------------------------------------------------------------
# كشف أمر /push
# -------------------------------------------------------------------

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
    words   = re.sub(r"[^\w\s]", "", instruction).split()
    slug    = "_".join(words[:4]).lower() if words else "script"
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


# -------------------------------------------------------------------
# دوال مشتركة
# -------------------------------------------------------------------

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


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------

def main():
    print("\n[Selfe Agent CI v5.1.0] تشغيل...")

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

    memory = MemoryManager(owner, repo, issue_number)

    try:
        from openai import OpenAI, RateLimitError, AuthenticationError
    except ImportError:
        write_output("⚠️ مكتبة openai غير مثبتَّتة.")
        sys.exit(1)

    cfg    = PROVIDER_CONFIG[model["provider"]]
    client = OpenAI(api_key=pkeys[0], base_url=cfg["base_url"])
    evaluator = SelfEvaluator(owner, repo, client, model["name"])

    # ── /push ───────────────────────────────────────────────
    is_push, push_instruction = detect_push_command(msg)

    if is_push:
        print(f"[CI] /push — {push_instruction}")
        filename = extract_filename(push_instruction)

        for attempt in range(MAX_RETRIES):
            try:
                resp = client.chat.completions.create(
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
                    f"✅ **تم دفع الملف بنجاح!**\n\n"
                    f"📄 **الملف:** `{filename}`\n"
                    f"🔗 **الـ commit:** {commit_url}\n\n"
                    f"```\n{clean_code[:1500]}\n```"
                )
                write_output(reply)
                memory.save_turn(msg, reply, model["name"], tokens, 0.2, True)
                return

            except RateLimitError:
                write_output("⚠️ Rate Limit. حاول لاحقاً.")
                sys.exit(1)
            except AuthenticationError:
                write_output("⚠️ مفتاح API غير صالح.")
                sys.exit(1)
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                else:
                    write_output(f"⚠️ خطأ /push: {e}")
                    sys.exit(1)

    # ── وضع عادي ───────────────────────────────────────────
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
            print(f"[SelfEval] 🔄 إعادة محاولة {eval_attempt}...")

        messages = memory.build_messages(current_system_prompt, msg)
        reply    = None
        tokens   = 0

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
                write_output("⚠️ Rate Limit.")
                sys.exit(1)
            except AuthenticationError:
                write_output("⚠️ Auth Error.")
                sys.exit(1)
            except Exception as e:
                if api_attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** api_attempt)
                else:
                    write_output(f"⚠️ خطأ: {e}")
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
        final_reply = best_reply or "⚠️ فشل الوكيل."

    if best_score < EVAL_THRESHOLD:
        final_reply += f"\n\n---\n> ⚠️ *أفضل تقييم ذاتي: {best_score}/10*"

    write_output(final_reply)
    memory.save_turn(msg, final_reply, model["name"], final_tokens, temp, True)
    print("[CI] اكتمل ✔")


if __name__ == "__main__":
    main()
