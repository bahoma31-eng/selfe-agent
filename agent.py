import os
import sys
import time

# =========================================================
# Selfe Agent v3.2.1
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
# =========================================================

import re
import urllib.request

MODELS_FILE        = "models.txt"
SYSTEM_PROMPT_FILE = "system_prompt.txt"
SKILLS_FILE        = "skills.txt"
MAX_RETRIES        = 3

DEFAULT_SYSTEM_PROMPT = "أنت مساعد ذكاء اصطناعي مفيد ودقيق."

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
    """
    تحديد قيمة temperature بناءً على نوع الطلب:
      - طلبات الكود والتصحيح → 0.2  (دقيق، أقل هلوسة)
      - طلبات حقيقية/تحليلية → 0.5  (توازن دقة/طلاقة)
      - طلبات إبداعية       → 0.9  (تنوع عالٍ)
      - عام                 → 0.7  (الافتراضي)
    """
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
    """
    قراءة skills.txt وإعادة قاموس {skill_name: url}.
    الصيغة: skill_name | raw_github_url
    """
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
    """
    جلب محتوى SKILL.md من raw GitHub URL.
    يستخدم urllib (بدون مكتبات خارجية).
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "selfe-agent/3.2"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8")
    except Exception as e:
        return ""


def activate_skill(skill_name: str, skills: dict[str, str],
                   history: list[dict]) -> tuple[str, list[dict]]:
    """
    تفعيل مهارة — جلب SKILL.md وضبطه كـ system message.
    تُعيد تاريخ المحادثة مع البرومبت الجديد.
    """
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
    """
    إرسال رسالة للنموذج مع:
      - temperature ديناميكية حسب نوع الطلب  [v3.1.0]
      - max_tokens=4096 لمنع قطع الردود      [v3.1.0]
      - retry مع Exponential Backoff         [v3.1.0]
    """
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
                break  # جرّب المفتاح التالي فوراً

            except AuthenticationError:
                print("[خطأ] مفتاح غير صالح — تحويل...")
                rotate_key(provider)
                break

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** attempt  # 1s، 2s، 4s
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
  جديد / new          — بدء محادثة جديدة
  تحديث / reload       — إعادة تحميل البرومبت من الملف
  خروج / exit          — إنهاء البرنامج
  مساعدة / help         — عرض هذه القائمة
""".strip()


def main():
    print("\n╔══════════════════════════════╗")
    print("║       Selfe Agent v3.2.1     ║")
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
