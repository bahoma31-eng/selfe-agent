import os
import sys

# =========================================================
# Selfe Agent — AI model runner
# المفتاح يُقرأ من متغير البيئة السري (GitHub Secrets)
# اسم النموذج يُقرأ من ملف models.txt
# =========================================================

MODELS_FILE = "models.txt"


def load_models(filepath: str) -> list[str]:
    """قراءة أسماء النماذج من الملف النصي (تجاهل التعليقات والأسطر الفارغة)."""
    if not os.path.exists(filepath):
        print(f"[ERROR] لم يُعثر على ملف النماذج: {filepath}")
        sys.exit(1)

    models = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                models.append(line)
    return models


def get_api_key() -> str:
    """قراءة مفتاح API من متغير البيئة السري OPENAI_API_KEY."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("[ERROR] لم يُعثر على المفتاح السري OPENAI_API_KEY في متغيرات البيئة.")
        print("       تأكد من إضافته في: Settings > Secrets and variables > Actions")
        sys.exit(1)
    return api_key


def select_model(models: list[str]) -> str:
    """عرض قائمة النماذج واختيار أحدها."""
    print("\n=== النماذج المتاحة ===")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")

    while True:
        try:
            choice = int(input("\nاختر رقم النموذج: "))
            if 1 <= choice <= len(models):
                return models[choice - 1]
            print(f"الرجاء إدخال رقم بين 1 و {len(models)}")
        except ValueError:
            print("الرجاء إدخال رقم صحيح.")


def chat(api_key: str, model: str, user_message: str) -> str:
    """إرسال رسالة إلى النموذج عبر OpenAI-compatible API وإعادة الرد."""
    try:
        from openai import OpenAI
    except ImportError:
        print("[ERROR] مكتبة openai غير مثبتة. نفّذ: pip install openai")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "أنت مساعد ذكاء اصطناعي مفيد ودقيق."},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content


def main():
    print("\n╔══════════════════════════════╗")
    print("║       Selfe Agent v1.0       ║")
    print("╚══════════════════════════════╝")

    # 1. تحميل المفتاح
    api_key = get_api_key()
    print("[OK] تم تحميل مفتاح API بنجاح.")

    # 2. تحميل قائمة النماذج
    models = load_models(MODELS_FILE)
    print(f"[OK] تم تحميل {len(models)} نموذج من {MODELS_FILE}")

    # 3. اختيار النموذج
    model = select_model(models)
    print(f"\n[OK] النموذج المختار: {model}")

    # 4. حلقة المحادثة
    print("\nاكتب رسالتك (اكتب 'خروج' للإنهاء):\n")
    while True:
        user_input = input("أنت: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("خروج", "exit", "quit"):
            print("وداعاً!")
            break

        print("النموذج: ", end="", flush=True)
        reply = chat(api_key, model, user_input)
        print(reply)
        print()


if __name__ == "__main__":
    main()
