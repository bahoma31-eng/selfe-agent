import os
import sys
import time
import subprocess
import textwrap
import datetime

# =========================================================
# Selfe Agent v3.3.1
# يدعم مفاتيح متعددة:
#   GEMINI_API_KEY_1 .. GEMINI_API_KEY_9  → نماذج Gemini
#   GROQ_API_KEY_1 .. GROQ_API_KEY_2      → نماذج Groq
# النماذج     → models.txt
# البرومبت    → system_prompt.txt  (افتراضي)
# المهارات    → skills.txt  (جلب SKILL.md من GitHub)
# =========================================================
# [v3.1.0] تحسينات المرحلة الرابعة:
#   - temperature ديناميكية حسب نوع الطلب
#   - max_tokens=4096 لمنع القطع
#   - آلية retry مع Exponential Backoff
# [v3.2.0] إضافة 5 مفاتيح API جديدة:
#   - Gemini: من 4 مفاتيح إلى 7 مفاتيح (GEMINI_API_KEY_5/6/7)
#   - Groq:   من مفتاح واحد إلى 2 مفاتيح مع تناوب تلقائي
# [v3.2.1] إضافة مفتاحَي Gemini الثامن والتاسع:
#   - GEMINI_API_KEY_8 و GEMINI_API_KEY_9
# [v3.3.0] تطبيق نمط autonomous-agent-patterns:
#   - أمر @run: توليد سكريبت Python تلقائياً وتنفيذه
#   - رفع السكريبت إلى scripts/ في المستودع
#   - تسجيل النتائج في reports/run_log.md
#   - استخدام متغيرات البيئة السرية تلقائياً
# [v3.3.1] إصلاح generate_script:
#   - حقن أسماء المتغيرات السرية الحقيقية في البرومبت
#   - النموذج يعرف الأسماء الصحيحة (SMTP_PASS لا SMTP_PASSWORD)
# =========================================================

import re
import urllib.request

MODELS_FILE        = "models.txt"
SYSTEM_PROMPT_FILE = "system_prompt.txt"
SKILLS_FILE        = "skills.txt"
SCRIPTS_DIR        = "scripts"
REPORTS_DIR        = "reports"
RUN_LOG_FILE       = os.path.join(REPORTS_DIR, "run_log.md")
MAX_RETRIES        = 3

DEFAULT_SYSTEM_PROMPT = "أنت مساعد ذكاء اصطناعي مفيد ودقيق."

# ── قائمة المتغيرات السرية المتاحة فعلاً في البيئة ──────────
# هذه هي الأسماء الحقيقية المضبوطة في GitHub Secrets
KNOWN_SECRET_VARS = [
    "SMTP_USER",        # عنوان البريد المُرسِل
    "SMTP_PASS",        # كلمة مرور التطبيق (App Password)
    "OWNER_EMAIL",      # بريد صاحب الوكيل
    "IMGBB_API_KEY",    # مفتاح رفع الصور على ImgBB
    "IG_USER_ID",       # معرّف حساب Instagram
    "PAT_TOKEN",        # Personal Access Token لـ GitHub
    "GROQ_API_KEY",     # مفتاح Groq (القديم للتوافق)
]

# ── إعدادات المزودين ───────────────────────────────────────
PROVIDER_CONFIG = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "secret_vars": ["GEMINI_API_KEY_1", "GEMINI_API_KEY_2",
                        "GEMINI_API_KEY_3", "GEMINI_API_KEY_4",
                        "GEMINI_API_KEY_5", "GEMINI_API_KEY_6",
                        "GEMINI_API_KEY_7", "GEMINI_API_KEY_8",
                        "GEMINI_API_KEY_9"],
        "rotate": True,
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "secret_vars": ["GROQ_API_KEY_1", "GROQ_API_KEY_2"],
        "rotate": True,
    },
}

_key_index: dict[str, int] = {p: 0 for p in PROVIDER_CONFIG}


# ─────────────────────────────────────────────────────────────
# temperature ديناميكية — [v3.1.0]
# ─────────────────────────────────────────────────────────────

def detect_temperature(message: str) -> float:
    msg = message.lower()
    code_keywords = [
        "كود", "code", "دالة", "function", "اكتب", "برمجة",
        "خطأ", "debug", "script", "سكريبت", "class", "كلاس",
        "api", "endpoint", "sql", "query", "استعلام",
    ]
    factual_keywords = [
        "ما هو", "what is", "كيف يعمل", "how does", "اشرح",
        "explain", "عرّف", "define", "فرق", "difference",
        "متى", "when", "لماذا", "why", "من هو", "who is",
    ]
    creative_keywords = [
        "قصيدة", "poem", "اقتراح", "suggest", "فكرة", "idea",
        "أفكار", "ideas", "إبداع", "creative", "تصميم", "design",
        "تخيّل", "imagine", "قصة", "story",
    ]
    if any(k in msg for k in code_keywords):
        return 0.2
    if any(k in msg for k in creative_keywords):
        return 0.9
    if any(k in msg for k in factual_keywords):
        return 0.5
    return 0.7


# ─────────────────────────────────────────────────────────────
# جلب SKILL.md من GitHub
# ─────────────────────────────────────────────────────────────

def load_skills(filepath: str) -> dict[str, str]:
    skills: dict[str, str] = {}
    if not os.path.exists(filepath):
        return skills
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                name, url = [p.strip() for p in line.split("|", 1)]
                skills[name.lower()] = url
    return skills


def fetch_skill(url: str) -> str:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "selfe-agent/3.3"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8")
    except Exception:
        return ""


def activate_skill(skill_name: str, skills: dict[str, str],
                   history: list[dict]) -> tuple[str, list[dict]]:
    key = skill_name.lower().strip()
    if key not in skills:
        available = ", ".join(skills.keys()) or "لا يوجد"
        return (
            f"[خطأ] المهارة '{skill_name}' غير موجودة.\n"
            f"المهارات المتاحة: {available}",
            history
        )
    url = skills[key]
    print(f"[جلب] جاري جلب SKILL.md من:\n  {url}")
    content = fetch_skill(url)
    if not content:
        return f"[خطأ] تعذّر جلب المهارة من الرابط.\nتحقّق من صحة الرابط في skills.txt", history
    new_history = [{"role": "system", "content": content}]
    msg = (
        f"[تم] تفعيل المهارة \033[1m{key}\033[0m ✔\n"
        f"  حجم المهارة: {len(content):,} حرف\n"
        f"  المحادثة السابقة أُعيدت تحضيرها بالسياق الجديد."
    )
    return msg, new_history


# ─────────────────────────────────────────────────────────────
# نمط autonomous-agent-patterns — [v3.3.0]
# إصلاح حقن المتغيرات — [v3.3.1]
# ─────────────────────────────────────────────────────────────

def build_secrets_context() -> str:
    """
    يبني نصاً يشرح للنموذج المتغيرات السرية المتاحة فعلاً،
    مع وصف دور كل متغير حتى يختار النموذج الصحيح منها.
    """
    lines = [
        "المتغيرات السرية المتاحة فعلاً في بيئة التنفيذ (استخدم os.environ.get بالاسم الصحيح):",
        "  - SMTP_USER      → عنوان البريد الإلكتروني المُرسِل (مثال: user@gmail.com)",
        "  - SMTP_PASS      → كلمة مرور التطبيق (App Password) لخادم SMTP",
        "  - OWNER_EMAIL    → البريد الإلكتروني لصاحب الوكيل (المستلم الافتراضي)",
        "  - IMGBB_API_KEY  → مفتاح API لرفع الصور على موقع ImgBB",
        "  - IG_USER_ID     → معرّف حساب Instagram",
        "  - PAT_TOKEN      → Personal Access Token للتعامل مع GitHub API",
        "  - GROQ_API_KEY   → مفتاح Groq API (للتوافق مع الإصدارات القديمة)",
        "",
        "ملاحظات مهمة:",
        "  • اسم متغير كلمة المرور هو SMTP_PASS وليس SMTP_PASSWORD",
        "  • خادم SMTP لـ Gmail: smtp.gmail.com | المنفذ: 587 | الأمان: STARTTLS",
        "  • استخدم smtplib.SMTP ثم .starttls() ثم .login(user, password)",
    ]
    return "\n".join(lines)


def generate_script(task_description: str, model_info: dict,
                    all_keys: dict) -> str:
    """
    يطلب من النموذج كتابة سكريبت Python لإنجاز المهمة.
    القواعد المُضمَّنة في البرومبت:
      1. يستخدم os.environ.get() بالأسماء الحقيقية للمتغيرات
      2. يطبع النتيجة النهائية في السطر الأخير
      3. يُعيد كود Python نظيفاً بدون markdown
    """
    secrets_context = build_secrets_context()

    prompt = textwrap.dedent(f"""
    أنت مساعد برمجي متخصص في كتابة سكريبتات Python.
    المهمة: {task_description}

    {secrets_context}

    اكتب سكريبت Python يُنجز هذه المهمة مع الالتزام بالقواعد التالية:
    1. استخدم os.environ.get("VARIABLE_NAME") بالأسماء الصحيحة المذكورة أعلاه فقط
    2. اطبع النتيجة النهائية بوضوح في نهاية السكريبت
    3. عالج الأخطاء باستخدام try/except وأظهر رسائل واضحة
    4. لا تضع أي شرح أو markdown — فقط كود Python صالح للتنفيذ مباشرة
    5. أضف تعليقاً في السطر الأول: # Task: {task_description[:60]}
    """).strip()

    history_tmp = [{"role": "system", "content": prompt}]
    raw = chat(model_info, all_keys, task_description, history_tmp)

    # استخراج الكود إذا جاء ضمن ```python ... ```
    code_match = re.search(r"```(?:python)?\n(.*?)```", raw, re.DOTALL)
    return code_match.group(1).strip() if code_match else raw.strip()


def save_script(script_code: str, task_name: str) -> str:
    """حفظ السكريبت في scripts/ وإعادة مساره."""
    os.makedirs(SCRIPTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^\w]", "_", task_name[:40]).lower()
    filename = f"{safe_name}_{timestamp}.py"
    filepath = os.path.join(SCRIPTS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(script_code)
    return filepath


def run_script(filepath: str) -> tuple[str, str, int]:
    """تنفيذ السكريبت وإعادة (stdout, stderr, returncode)."""
    try:
        result = subprocess.run(
            [sys.executable, filepath],
            capture_output=True,
            text=True,
            timeout=60,
            env=os.environ.copy(),
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "[خطأ] انتهت مهلة التنفيذ (60 ثانية).", 1
    except Exception as e:
        return "", f"[خطأ] فشل التنفيذ: {e}", 1


def log_run(task: str, script_path: str,
            stdout: str, stderr: str, returncode: int) -> None:
    """تسجيل نتيجة التنفيذ في reports/run_log.md."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "✅ نجح" if returncode == 0 else "❌ فشل"
    entry = textwrap.dedent(f"""
    ---
    ## {now} — {status}
    **المهمة:** {task}
    **السكريبت:** `{script_path}`
    **كود الخروج:** `{returncode}`

    ### المخرجات
    ```
    {stdout.strip() or '(لا مخرجات)'}
    ```

    ### الأخطاء
    ```
    {stderr.strip() or '(لا أخطاء)'}
    ```
    """).strip() + "\n"

    mode = "a" if os.path.exists(RUN_LOG_FILE) else "w"
    with open(RUN_LOG_FILE, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write("# سجل تشغيل السكريبتات — Selfe Agent\n\n")
        f.write(entry + "\n")


def handle_run_command(task: str, model_info: dict, all_keys: dict) -> str:
    """
    تنفيذ نمط autonomous-agent-patterns:
      1. توليد السكريبت بالذكاء الاصطناعي
      2. حفظه في scripts/
      3. تنفيذه
      4. تسجيل النتيجة في reports/run_log.md
      5. إعادة تقرير موجز للمستخدم
    """
    print(f"\n[🤖 @run] جاري توليد السكريبت للمهمة: {task}")
    script_code = generate_script(task, model_info, all_keys)

    script_path = save_script(script_code, task)
    print(f"[💾] السكريبت محفوظ: {script_path}")
    print(f"[▶️ ] جاري التنفيذ...")

    stdout, stderr, rc = run_script(script_path)
    log_run(task, script_path, stdout, stderr, rc)

    status = "✅ نجح" if rc == 0 else "❌ فشل"
    report = (
        f"\n{'='*50}\n"
        f"[{status}] نتيجة تنفيذ: {task}\n"
        f"السكريبت: {script_path}\n"
        f"{'='*50}\n"
    )
    if stdout.strip():
        report += f"📤 المخرجات:\n{stdout.strip()}\n"
    if stderr.strip():
        report += f"⚠️  الأخطاء:\n{stderr.strip()}\n"
    report += f"\n📋 التقرير الكامل محفوظ في: {RUN_LOG_FILE}"
    return report


# ─────────────────────────────────────────────────────────────
# بقية الدوال (keys, models, prompt, chat)
# ─────────────────────────────────────────────────────────────

def load_system_prompt(filepath: str) -> str:
    if not os.path.exists(filepath):
        return DEFAULT_SYSTEM_PROMPT
    lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip().startswith("#"):
                lines.append(line.rstrip())
    prompt = "\n".join(lines).strip()
    return prompt or DEFAULT_SYSTEM_PROMPT


def load_keys() -> dict[str, list[str]]:
    keys: dict[str, list[str]] = {}
    missing: list[str] = []
    for provider, cfg in PROVIDER_CONFIG.items():
        pkeys = []
        for var in cfg["secret_vars"]:
            val = os.environ.get(var, "").strip()
            if val:
                pkeys.append(val)
            else:
                missing.append(var)
        keys[provider] = pkeys
    if missing:
        print("[تحذير] متغيرات غير موجودة:")
        for m in missing:
            print(f"  - {m}")
        print()
    for provider, pkeys in keys.items():
        if not pkeys:
            print(f"[ERROR] لا يوجد مفتاح صالح للمزود '{provider}'.")
            sys.exit(1)
    return keys


def get_key(provider: str, all_keys: dict) -> str:
    pkeys = all_keys[provider]
    return pkeys[_key_index[provider] % len(pkeys)]


def rotate_key(provider: str) -> None:
    _key_index[provider] += 1


def load_models(filepath: str) -> list[dict]:
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
            provider = provider.lower()
            if provider not in PROVIDER_CONFIG:
                continue
            models.append({"name": name, "provider": provider})
    return models


def select_model(models: list[dict]) -> dict:
    print("\n╔══════╦═══════════════════════════╦═══════════╗")
    print("║  #   ║  اسم النموذج                  ║  المزود   ║")
    print("╠══════╬═══════════════════════════╬═══════════╣")
    for i, m in enumerate(models, 1):
        print(f"║  {i:<3} ║  {m['name']:<29} ║  {m['provider']:<9} ║")
    print("╚══════╩═══════════════════════════╩═══════════╝")
    while True:
        try:
            choice = int(input("\nاختر رقم النموذج: "))
            if 1 <= choice <= len(models):
                return models[choice - 1]
        except ValueError:
            pass
        print(f"رقم غير صحيح. أدخل بين 1 و {len(models)}.")


def chat(model_info: dict, all_keys: dict,
         user_message: str, history: list[dict]) -> str:
    try:
        from openai import OpenAI, RateLimitError, AuthenticationError
    except ImportError:
        print("[ERROR] pip install openai")
        sys.exit(1)

    provider  = model_info["provider"]
    cfg       = PROVIDER_CONFIG[provider]
    pkeys     = all_keys[provider]
    messages  = history + [{"role": "user", "content": user_message}]
    temp      = detect_temperature(user_message)

    for _ in range(len(pkeys)):
        client = OpenAI(api_key=get_key(provider, all_keys),
                        base_url=cfg["base_url"])

        for attempt in range(MAX_RETRIES):
            try:
                resp = client.chat.completions.create(
                    model=model_info["name"],
                    messages=messages,
                    temperature=temp,
                    max_tokens=4096,
                )
                return resp.choices[0].message.content

            except RateLimitError:
                print("[تحذير] Rate limit — تحويل المفتاح...")
                rotate_key(provider)
                break

            except AuthenticationError:
                print("[خطأ] مفتاح غير صالح — تحويل...")
                rotate_key(provider)
                break

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    print(f"[تحذير] خطأ مؤقت، إعادة المحاولة "
                          f"{attempt + 1}/{MAX_RETRIES} بعد {wait}s...")
                    time.sleep(wait)
                else:
                    return f"[خطأ] {e}"

    return "[خطأ] جميع المفاتيح استُنفدت."


# ─────────────────────────────────────────────────────────────
# البرنامج الرئيسي
# ─────────────────────────────────────────────────────────────

HELP_TEXT = """
أوامر متاحة:
  @skill <اسم>          — تفعيل مهارة (تجلب SKILL.md وتضبطه كبرومبت)
  @skill <اسم> <رسالة>  — تفعيل المهارة وإرسال رسالة فوراً
  @skills              — عرض المهارات المتاحة
  @run <وصف المهمة>    — توليد سكريبت Python وتنفيذه تلقائياً
  جديد / new          — بدء محادثة جديدة
  تحديث / reload       — إعادة تحميل البرومبت من الملف
  خروج / exit          — إنهاء البرنامج
  مساعدة / help         — عرض هذه القائمة
""".strip()


def main():
    print("\n╔══════════════════════════════╗")
    print("║       Selfe Agent v3.3.1     ║")
    print("╚══════════════════════════════╝")

    # 1. تحميل المهارات
    skills = load_skills(SKILLS_FILE)
    if skills:
        print(f"[OK] المهارات المتاحة: {', '.join(skills.keys())}")
    else:
        print("[تحذير] لم يُعثر على skills.txt أو فارغ.")

    # 2. تحميل البرومبت الافتراضي
    system_prompt = load_system_prompt(SYSTEM_PROMPT_FILE)
    preview = system_prompt[:80].replace("\n", " ")
    print(f"[OK] البرومبت: \"{preview}{'...' if len(system_prompt) > 80 else ''}\"")

    # 3. تحميل المفاتيح
    all_keys = load_keys()
    for provider, pkeys in all_keys.items():
        print(f"[OK] {provider}: {len(pkeys)} مفتاح محمّل.")

    # 4. تحميل النماذج
    models = load_models(MODELS_FILE)
    print(f"[OK] تم تحميل {len(models)} نموذج.")

    # 5. اختيار النموذج
    model_info = select_model(models)
    print(f"\n[OK] النموذج : {model_info['name']}  |المزود: {model_info['provider']}")
    print("\nاكتب 'مساعدة' لعرض جميع الأوامر.\n")

    # 6. حلقة المحادثة
    history: list[dict] = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            user_input = input("أنت: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nوداعاً!")
            break

        if not user_input:
            continue

        low = user_input.lower()

        # ── أوامر النظام ─────────────────────────────────
        if low in ("خروج", "exit", "quit"):
            print("وداعاً!")
            break

        if low in ("مساعدة", "help"):
            print(HELP_TEXT)
            continue

        if low in ("جديد", "new", "reset"):
            history = [{"role": "system", "content": history[0]["content"]}]
            print("[تم] بدء محادثة جديدة.\n")
            continue

        if low in ("تحديث", "reload"):
            system_prompt = load_system_prompt(SYSTEM_PROMPT_FILE)
            history[0]["content"] = system_prompt
            print("[تم] إعادة تحميل البرومبت.\n")
            continue

        if low == "@skills":
            if skills:
                print("المهارات المتاحة:")
                for name, url in skills.items():
                    print(f"  - {name}\n    {url}")
            else:
                print("لا توجد مهارات. أضف مهارات في skills.txt")
            print()
            continue

        # ── أمر @run — autonomous-agent-patterns [v3.3.0] ──
        if low.startswith("@run "):
            task = user_input[len("@run "):].strip()
            if not task:
                print("[خطأ] أدخل وصف المهمة بعد @run\n")
                continue
            result = handle_run_command(task, model_info, all_keys)
            print(result)
            print()
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": result})
            continue

        # ── أمر @skill ───────────────────────────────────
        if low.startswith("@skill "):
            rest = user_input[len("@skill "):].strip()
            parts = rest.split(None, 1)
            skill_name  = parts[0] if parts else ""
            inline_msg  = parts[1] if len(parts) > 1 else None

            result_msg, history = activate_skill(skill_name, skills, history)
            print(result_msg)
            print()

            if inline_msg and "[خطأ]" not in result_msg:
                print(f"{model_info['name']}: ", end="", flush=True)
                reply = chat(model_info, all_keys, inline_msg, history)
                print(reply)
                print()
                history.append({"role": "user",      "content": inline_msg})
                history.append({"role": "assistant", "content": reply})
            continue

        # ── رسالة عادية ───────────────────────────────────
        print(f"{model_info['name']}: ", end="", flush=True)
        reply = chat(model_info, all_keys, user_input, history)
        print(reply)
        print()
        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
