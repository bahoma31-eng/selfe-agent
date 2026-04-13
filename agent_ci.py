# =============================================================
# agent_ci.py — Selfe Agent v3.1.0 (نسخة GitHub Actions)
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


# ────────────────────────────────────────────────────────────────
de load_system_prompt(filepath: str) -> str:
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
    """
    temperature ديناميكية حسب نوع الطلب — مستوردة من agent.py
    """
    msg = message.lower()
    code_kw     = ["\u0643\u0648\u062f","code","\u062f\u0627\u0644\u0629","function","\u0627\u0643\u062a\u0628","\u0628\u0631\u0645\u062c\u0629",
                   "\u062e\u0637\u0623","debug","script","\u0633\u0643\u0631\u064a\u0628\u062a","class","\u0643\u0644\u0627\u0633",
                   "api","endpoint","sql","query","\u0627\u0633\u062a\u0639\u0644\u0627\u0645"]
    factual_kw  = ["\u0645\u0627 \u0647\u0648","what is","\u0643\u064a\u0641 \u064a\u0639\u0645\u0644","how does","\u0627\u0634\u0631\u062d",
                   "explain","\u0639\u0631\u0651\u0641","define","\u0641\u0631\u0642","difference",
                   "\u0645\u062a\u0649","when","\u0644\u0645\u0627\u0630\u0627","why","\u0645\u0646 \u0647\u0648","who is"]
    creative_kw = ["\u0642\u0635\u064a\u062f\u0629","poem","\u0627\u0642\u062a\u0631\u0627\u062d","suggest","\u0641\u0643\u0631\u0629","idea",
                   "\u0623\u0641\u0643\u0627\u0631","ideas","\u0625\u0628\u062f\u0627\u0639","creative","\u062a\u0635\u0645\u064a\u0645","design",
                   "\u062a\u062e\u064a\u0651\u0644","imagine","\u0642\u0635\u0629","story"]
    if any(k in msg for k in code_kw):     return 0.2
    if any(k in msg for k in creative_kw): return 0.9
    if any(k in msg for k in factual_kw):  return 0.5
    return 0.7


def write_output(reply: str) -> None:
    """\u0643\u062a\u0627\u0628\u0629 \u0627\u0644\u0631\u062f \u0641\u064a GITHUB_OUTPUT \u0644\u064a\u064f\u0642\u0631\u0623\u0647 \u0627\u0644\u0640 workflow و\u064a\u0646\u0634\u0631\u0647 \u0643\u0640 comment"""
    gh_output = os.environ.get("GITHUB_OUTPUT", "")
    if gh_output:
        with open(gh_output, "a", encoding="utf-8") as fh:
            fh.write(f"reply<<EOF\n{reply}\nEOF\n")
    print("[Selfe Reply]")
    print(reply)


def main():
    print("\n[Selfe Agent CI] \u062a\u0634\u063a\u064a\u0644 \u0627\u0644\u0648\u0643\u064a\u0644 \u0641\u064a \u0628\u064a\u0626\u0629 GitHub Actions...")

    # ─ ق\u0631\u0627\u0621\u0629 \u0627\u0644\u0631\u0633\u0627\u0644\u0629 \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    msg = os.environ.get("USER_MESSAGE", "").strip()
    if not msg:
        write_output("⚠️ لم يتم استقبال أي رسالة من الـ Issue.")
        sys.exit(0)

    # ─ ت\u062d\u0645\u064a\u0644 \u0627\u0644\u0646\u0645\u0627\u0630\u062c \u0648\u0627\u062e\u062a\u064a\u0627\u0631 \u0627\u0644\u0645\u0637\u0644\u0648\u0628 \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    models = load_models(MODELS_FILE)
    idx    = int(os.environ.get("MODEL_INDEX", "1")) - 1
    model  = models[idx] if idx < len(models) else models[0]
    print(f"[CI] \u0627\u0644\u0646\u0645\u0648\u0630\u062c: {model['name']} | \u0627\u0644\u0645\u0632\u0648\u062f: {model['provider']}")

    # ─ ت\u062d\u0645\u064a\u0644 \u0645\u0641\u0627\u062a\u064a\u062d API \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
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

    # ─ ت\u062d\u0645\u064a\u0644 \u0627\u0644\u0628\u0631\u0648\u0645\u0628\u062a \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    system_prompt = load_system_prompt(SYSTEM_PROMPT_FILE)
    temp          = detect_temperature(msg)
    print(f"[CI] Temperature: {temp}")

    # ─ \u0627\u0644\u0627\u062a\u0635\u0627\u0644 \u0628\u0627\u0644\u0646\u0645\u0648\u0630\u062c \u0645\u0639 Retry \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
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
