# =============================================================
# agent_ci.py — Selfe Agent v3.1.1 (نسخة GitHub Actions)
# تعمل بدون تفاعل (non-interactive)
# تقرأ USER_MESSAGE و MODEL_INDEX من environment variables
# تكتب الرد في GITHUB_OUTPUT ليُنشر كـ comment على الـ Issue
# =============================================================

import os
import sys
import time

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
                   "api", "endpoint", "sql", "query", "استعلام"]
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


def main():
    print("\n[Selfe Agent CI] تشغيل الوكيل في بيئة GitHub Actions...")

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
    print(f"[CI] عدد المفاتيح: {len(pkeys)}")

    system_prompt = load_system_prompt(SYSTEM_PROMPT_FILE)
    temp          = detect_temperature(msg)
    print(f"[CI] Temperature: {temp}")

    try:
        from openai import OpenAI, RateLimitError, AuthenticationError
    except ImportError:
        write_output("⚠️ مكتبة openai غير مثبتَّتة.")
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
