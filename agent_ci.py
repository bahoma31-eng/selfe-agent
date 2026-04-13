# =============================================================
# agent_ci.py — Selfe Agent v3.2.0 (نسخة GitHub Actions)
# تعمل بدون تفاعل (non-interactive)
# تقرأ USER_MESSAGE و MODEL_INDEX من environment variables
# تكتب الرد في GITHUB_OUTPUT ليُنشر كـ comment على الـ Issue
#
# ميزة /push:
#   اكتب في الـ Issue أو Comment:
#   /push اكتب سكريبت Python يحسب الفيبوناتشي واحفظه في fibonacci.py
#   سيقوم الوكيل بكتابة الكود ودفعه مباشرةً إلى main
# =============================================================

import os
import sys
import time
import re
import json
import base64
import urllib.request
import urllib.error

MODELS_FILE        = "models.txt"
SYSTEM_PROMPT_FILE = "system_prompt.txt"
MAX_RETRIES        = 3

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
        "User-Agent": "selfe-agent/3.2.0",
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
    مستوحى من مهارة github-automation — GITHUB_GET_REPOSITORY_CONTENT + PUT /contents
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
    """
    إرجاع (True, instruction) إذا وُجد /push، وإلا (False, '').
    """
    m = PUSH_PATTERN.search(message)
    if m:
        return True, m.group(1).strip()
    return False, ""


def extract_filename(instruction: str) -> str:
    """استخراج اسم الملف من التعليمة، مع fallback ذكي."""
    m = FILE_NAME_PATTERN.search(instruction)
    if m:
        return m.group(1)
    # fallback: توليد اسم من الكلمات الأولى
    words = re.sub(r"[^\w\s]", "", instruction).split()
    slug = "_".join(words[:4]).lower() if words else "script"
    # تخمين اللغة
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
    """استخراج الكود النظيف من رد الـ LLM (بدون code fences)."""
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
# PUSH SYSTEM PROMPT — يطلب كوداً نظيفاً فقط
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
    print("\n[Selfe Agent CI v3.2.0] تشغيل الوكيل في بيئة GitHub Actions...")

    msg = os.environ.get("USER_MESSAGE", "").strip()
    if not msg:
        write_output("⚠️ لم يتم استقبال أي رسالة من الـ Issue.")
        sys.exit(0)

    models = load_models(MODELS_FILE)
    idx    = int(os.environ.get("MODEL_INDEX", "1")) - 1
    model  = models[idx] if idx < len(models) else models[0]
    print(f"[CI] النموذج: {model['name']} | المزود: {model['provider']}")

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
                raw_reply = resp.choices[0].message.content
                clean_code = extract_code_from_reply(raw_reply)

                # استخراج بيانات المستودع من GITHUB_REPOSITORY
                gh_repo = os.environ.get("GITHUB_REPOSITORY", "/")
                owner, repo = gh_repo.split("/", 1)

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
                print("[CI] /push اكتمل بنجاح ✔")
                return

            except RateLimitError:
                write_output("⚠️ تجاوز حد الطلبات (Rate Limit). حاول لاحقاً.")
                sys.exit(1)
            except AuthenticationError:
                write_output("⚠️ مفتاح API غير صالح. تحقّق من GitHub Secrets.")
                sys.exit(1)
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    print(f"[تحذير] خطأ، إعادة المحاولة {attempt+1}/{MAX_RETRIES} بعد {wait}s...")
                    time.sleep(wait)
                else:
                    write_output(f"⚠️ خطأ أثناء /push: {e}")
                    sys.exit(1)

    # ---------------------------------------------------------------
    # الوضع العادي — رد نصي على الـ Issue
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

    for attempt in range(MAX_RETRIES):
        try:
            client = OpenAI(api_key=pkeys[0], base_url=cfg["base_url"])
            resp   = client.chat.completions.create(
                model=model["name"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": msg},
                ],
                temperature=temp,
                max_tokens=4096,
            )
            reply = resp.choices[0].message.content
            write_output(reply)
            print("[CI] اكتمل بنجاح ✔")
            return

        except RateLimitError:
            write_output("⚠️ تجاوز حد الطلبات (Rate Limit). حاول لاحقاً.")
            sys.exit(1)
        except AuthenticationError:
            write_output("⚠️ مفتاح API غير صالح. تحقّق من GitHub Secrets.")
            sys.exit(1)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"[تحذير] خطأ مؤقت، إعادة المحاولة {attempt+1}/{MAX_RETRIES} بعد {wait}s...")
                time.sleep(wait)
            else:
                write_output(f"⚠️ خطأ: {e}")
                sys.exit(1)


if __name__ == "__main__":
    main()
