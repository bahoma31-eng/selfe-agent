import os
import sys

# =========================================================
# Selfe Agent v2.1
# يدعم مفاتيح متعددة:
#   GEMINI_API_KEY_1 .. GEMINI_API_KEY_4  → نماذج Gemini
#   GROQ_API_KEY                          → نماذج Groq
# أسماء النماذج + المزود  →  models.txt
# البرومبت الخاص بالوكيل  →  system_prompt.txt
# =========================================================

MODELS_FILE        = "models.txt"
SYSTEM_PROMPT_FILE = "system_prompt.txt"

DEFAULT_SYSTEM_PROMPT = "أنت مساعد ذكاء اصطناعي مفيد ودقيق."

# ── إعدادات المزودين ───────────────────────────────────────
PROVIDER_CONFIG = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "secret_vars": ["GEMINI_API_KEY_1", "GEMINI_API_KEY_2",
                        "GEMINI_API_KEY_3", "GEMINI_API_KEY_4"],
        "rotate": True,
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "secret_vars": ["GROQ_API_KEY"],
        "rotate": False,
    },
}

_key_index: dict[str, int] = {p: 0 for p in PROVIDER_CONFIG}


# ── تحميل البرومبت ─────────────────────────────────────────
def load_system_prompt(filepath: str) -> str:
    """
    قراءة البرومبت من system_prompt.txt.
    - تُتجاهل الأسطر الفارغة والتعليقات (#).
    - إذا لم يوجد الملف، يستخدم البرومبت الافتراضي.
    """
    if not os.path.exists(filepath):
        print(f"[تحذير] لم يُعثر على {filepath} — سيستخدم البرومبت الافتراضي.")
        return DEFAULT_SYSTEM_PROMPT

    lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            # تخطّي التعليقات
            if stripped.startswith("#"):
                continue
            lines.append(line.rstrip())

    prompt = "\n".join(lines).strip()

    if not prompt:
        print(f"[تحذير] الملف {filepath} فارغ — سيستخدم البرومبت الافتراضي.")
        return DEFAULT_SYSTEM_PROMPT

    return prompt


# ── تحميل المفاتيح ─────────────────────────────────────────
def load_keys() -> dict[str, list[str]]:
    keys: dict[str, list[str]] = {}
    missing: list[str] = []

    for provider, cfg in PROVIDER_CONFIG.items():
        provider_keys = []
        for var in cfg["secret_vars"]:
            val = os.environ.get(var, "").strip()
            if val:
                provider_keys.append(val)
            else:
                missing.append(var)
        keys[provider] = provider_keys

    if missing:
        print("[تحذير] المتغيرات التالية غير موجودة أو فارغة:")
        for m in missing:
            print(f"         - {m}")
        print("  أضفها في: Settings → Secrets and variables → Actions\n")

    for provider, pkeys in keys.items():
        if not pkeys:
            print(f"[ERROR] لا يوجد أي مفتاح صالح للمزود '{provider}'.")
            sys.exit(1)

    return keys


def get_key(provider: str, all_keys: dict[str, list[str]]) -> str:
    pkeys = all_keys[provider]
    return pkeys[_key_index[provider] % len(pkeys)]


def rotate_key(provider: str) -> None:
    _key_index[provider] += 1


# ── تحميل النماذج ───────────────────────────────────────────
def load_models(filepath: str) -> list[dict]:
    if not os.path.exists(filepath):
        print(f"[ERROR] لم يُعثر على ملف النماذج: {filepath}")
        sys.exit(1)

    models = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                name, provider = [p.strip() for p in line.split("|", 1)]
            else:
                name, provider = line, "gemini"
            provider = provider.lower()
            if provider not in PROVIDER_CONFIG:
                print(f"[تحذير] مزود غير معروف '{provider}' — تم تخطيه.")
                continue
            models.append({"name": name, "provider": provider})
    return models


# ── اختيار النموذج ──────────────────────────────────────────
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
            print(f"الرجاء إدخال رقم بين 1 و {len(models)}")
        except ValueError:
            print("الرجاء إدخال رقم صحيح.")


# ── إرسال الرسالة ───────────────────────────────────────────
def chat(model_info: dict, all_keys: dict[str, list[str]],
        user_message: str, history: list[dict]) -> str:
    try:
        from openai import OpenAI, RateLimitError, AuthenticationError
    except ImportError:
        print("[ERROR] مكتبة openai غير مثبتة. نفّذ: pip install openai")
        sys.exit(1)

    provider = model_info["provider"]
    cfg      = PROVIDER_CONFIG[provider]
    pkeys    = all_keys[provider]
    messages = history + [{"role": "user", "content": user_message}]

    for _ in range(len(pkeys)):
        client = OpenAI(api_key=get_key(provider, all_keys),
                        base_url=cfg["base_url"])
        try:
            resp = client.chat.completions.create(
                model=model_info["name"],
                messages=messages,
                temperature=0.7,
            )
            return resp.choices[0].message.content

        except RateLimitError:
            print(f"[تحذير] حد الاستخدام — تحويل المفتاح...")
            rotate_key(provider)
        except AuthenticationError:
            print(f"[خطأ] مفتاح غير صالح — تحويل...")
            rotate_key(provider)
        except Exception as e:
            return f"[خطأ غير متوقع] {e}"

    return "[خطأ] جميع المفاتيح استُنفدت أو غير صالحة."


# ── البرنامج الرئيسي ────────────────────────────────────────
def main():
    print("\n╔══════════════════════════════╗")
    print("║       Selfe Agent v2.1       ║")
    print("╚══════════════════════════════╝")

    # 1. تحميل البرومبت
    system_prompt = load_system_prompt(SYSTEM_PROMPT_FILE)
    # عرض معاينة مختصرة للبرومبت
    preview = system_prompt[:80].replace("\n", " ")
    print(f"[OK] البرومبت: \"{preview}{'...' if len(system_prompt) > 80 else ''}\"")

    # 2. تحميل المفاتيح
    all_keys = load_keys()
    for provider, pkeys in all_keys.items():
        print(f"[OK] {provider}: {len(pkeys)} مفتاح محمّل.")

    # 3. تحميل النماذج
    models = load_models(MODELS_FILE)
    print(f"[OK] تم تحميل {len(models)} نموذج من {MODELS_FILE}")

    # 4. اختيار النموذج
    model_info = select_model(models)
    print(f"\n[OK] النموذج المختار : {model_info['name']}")
    print(f"[OK] المزود          : {model_info['provider']}")

    # 5. حلقة المحادثة
    print("\nاكتب رسالتك (اكتب 'خروج' للإنهاء، 'جديد' لمحادثة جديدة):\n")

    # التاريخ يبدأ بالبرومبت المقروء من الملف
    history: list[dict] = [
        {"role": "system", "content": system_prompt}
    ]

    while True:
        user_input = input("أنت: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("خروج", "exit", "quit"):
            print("وداعاً!")
            break
        if user_input.lower() in ("جديد", "new", "reset"):
            history = [{"role": "system", "content": system_prompt}]
            print("[تم] بدء محادثة جديدة.\n")
            continue
        # إعادة تحميل البرومبت لحظياً بدون إعادة تشغيل
        if user_input.lower() in ("تحديث", "reload"):
            system_prompt = load_system_prompt(SYSTEM_PROMPT_FILE)
            history[0]["content"] = system_prompt
            print("[تم] إعادة تحميل البرومبت.\n")
            continue

        print(f"{model_info['name']}: ", end="", flush=True)
        reply = chat(model_info, all_keys, user_input, history)
        print(reply)
        print()

        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
