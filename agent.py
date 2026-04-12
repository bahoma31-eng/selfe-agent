import os
import sys

# =========================================================
# Selfe Agent v2.0
# يدعم مفاتيح متعددة:
#   GEMINI_API_KEY_1 .. GEMINI_API_KEY_4  → نماذج Gemini
#   GROQ_API_KEY                          → نماذج Groq
# أسماء النماذج + المزود تُقرأ من models.txt
# =========================================================

MODELS_FILE = "models.txt"

# ── إعدادات المزودين ───────────────────────────────────────
PROVIDER_CONFIG = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "secret_vars": ["GEMINI_API_KEY_1", "GEMINI_API_KEY_2",
                        "GEMINI_API_KEY_3", "GEMINI_API_KEY_4"],
        "rotate": True,   # يدور على المفاتيح تلقائياً عند الخطأ
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "secret_vars": ["GROQ_API_KEY"],
        "rotate": False,
    },
}

# مؤشر التدوير لكل مزود (يرتفع عند rate-limit أو خطأ في المفتاح)
_key_index: dict[str, int] = {p: 0 for p in PROVIDER_CONFIG}


# ── تحميل المفاتيح ──────────────────────────────────────────
def load_keys() -> dict[str, list[str]]:
    """قراءة جميع المفاتيح من متغيرات البيئة وإعادتها مجمّعة حسب المزود."""
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
        print("[تحذير] المتغيرات التالية غير موجودة أو فارغة في البيئة:")
        for m in missing:
            print(f"         - {m}")
        print("  أضفها في: Settings → Secrets and variables → Actions\n")

    # تحقق أن كل مزود لديه مفتاح واحد على الأقل
    for provider, pkeys in keys.items():
        if not pkeys:
            print(f"[ERROR] لا يوجد أي مفتاح صالح للمزود '{provider}'.")
            print(f"        أضف على الأقل المتغير: {PROVIDER_CONFIG[provider]['secret_vars'][0]}")
            sys.exit(1)

    return keys


def get_key(provider: str, all_keys: dict[str, list[str]]) -> str:
    """إعادة المفتاح الحالي للمزود (مع دعم التدوير)."""
    pkeys = all_keys[provider]
    idx = _key_index[provider] % len(pkeys)
    return pkeys[idx]


def rotate_key(provider: str) -> None:
    """الانتقال إلى المفتاح التالي للمزود (عند rate-limit)."""
    _key_index[provider] += 1


# ── تحميل النماذج ───────────────────────────────────────────
def load_models(filepath: str) -> list[dict]:
    """قراءة النماذج من models.txt بصيغة 'model_name | provider'."""
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
                name, provider = line, "gemini"  # افتراضي
            provider = provider.lower()
            if provider not in PROVIDER_CONFIG:
                print(f"[تحذير] مزود غير معروف '{provider}' للنموذج '{name}' — تم تخطيه.")
                continue
            models.append({"name": name, "provider": provider})
    return models


# ── اختيار النموذج ──────────────────────────────────────────
def select_model(models: list[dict]) -> dict:
    print("\n╔══════════════════════════════════════════════════╗")
    print("║            النماذج المتاحة                      ║")
    print("╠══════╦═══════════════════════════════╦═══════════╣")
    print("║  #   ║  اسم النموذج                  ║  المزود   ║")
    print("╠══════╬═══════════════════════════════╬═══════════╣")
    for i, m in enumerate(models, 1):
        print(f"║  {i:<3} ║  {m['name']:<29} ║  {m['provider']:<9} ║")
    print("╚══════╩═══════════════════════════════╩═══════════╝")

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
    """إرسال رسالة والحصول على رد مع دعم التدوير التلقائي للمفاتيح."""
    try:
        from openai import OpenAI, RateLimitError, AuthenticationError
    except ImportError:
        print("[ERROR] مكتبة openai غير مثبتة. نفّذ: pip install openai")
        sys.exit(1)

    provider = model_info["provider"]
    cfg = PROVIDER_CONFIG[provider]
    pkeys = all_keys[provider]
    max_attempts = len(pkeys)

    messages = history + [{"role": "user", "content": user_message}]

    for attempt in range(max_attempts):
        api_key = get_key(provider, all_keys)
        client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
        try:
            response = client.chat.completions.create(
                model=model_info["name"],
                messages=messages,
                temperature=0.7,
            )
            return response.choices[0].message.content

        except RateLimitError:
            print(f"[تحذير] المفتاح {_key_index[provider] % len(pkeys) + 1} وصل الحد — "
                  f"التبديل إلى المفتاح التالي...")
            rotate_key(provider)

        except AuthenticationError:
            print(f"[خطأ] مفتاح غير صالح ({cfg['secret_vars'][_key_index[provider] % len(pkeys)]}) — "
                  f"التبديل...")
            rotate_key(provider)

        except Exception as e:
            return f"[خطأ غير متوقع] {e}"

    return "[خطأ] جميع المفاتيح المتاحة استُنفدت أو غير صالحة."


# ── البرنامج الرئيسي ────────────────────────────────────────
def main():
    print("\n╔══════════════════════════════╗")
    print("║       Selfe Agent v2.0       ║")
    print("╚══════════════════════════════╝")

    # 1. تحميل المفاتيح
    all_keys = load_keys()
    for provider, pkeys in all_keys.items():
        print(f"[OK] {provider}: {len(pkeys)} مفتاح محمّل.")

    # 2. تحميل النماذج
    models = load_models(MODELS_FILE)
    print(f"[OK] تم تحميل {len(models)} نموذج من {MODELS_FILE}")

    # 3. اختيار النموذج
    model_info = select_model(models)
    print(f"\n[OK] النموذج المختار : {model_info['name']}")
    print(f"[OK] المزود          : {model_info['provider']}")

    # 4. حلقة المحادثة مع حفظ السياق
    print("\nاكتب رسالتك (اكتب 'خروج' للإنهاء، 'جديد' لمحادثة جديدة):\n")
    history: list[dict] = [
        {"role": "system", "content": "أنت مساعد ذكاء اصطناعي مفيد ودقيق."}
    ]

    while True:
        user_input = input("أنت: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("خروج", "exit", "quit"):
            print("وداعاً!")
            break
        if user_input.lower() in ("جديد", "new", "reset"):
            history = [{"role": "system", "content": "أنت مساعد ذكاء اصطناعي مفيد ودقيق."}]
            print("[تم] بدء محادثة جديدة.\n")
            continue

        print(f"{model_info['name']}: ", end="", flush=True)
        reply = chat(model_info, all_keys, user_input, history)
        print(reply)
        print()

        # حفظ السياق
        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
