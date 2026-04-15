import os
import json
from groq import Groq

def load_available_secrets():
    """تحميل قائمة الـ secrets المتاحة فعلاً في البيئة"""
    secrets_file = os.environ.get('AVAILABLE_SECRETS_FILE', 'available_secrets.txt')
    available = []
    if os.path.exists(secrets_file):
        with open(secrets_file) as f:
            available = [line.strip() for line in f if line.strip()]
    return available

def run_planner(issue_number: int, issue_title: str, issue_body: str):
    client = Groq(api_key=os.environ['GROQ_API_KEY'])

    available_secrets = load_available_secrets()
    secrets_info = "\n".join(f"- {s}" for s in available_secrets) if available_secrets else "- GROQ_API_KEY"

    prompt = f"""You are a precise task planning agent for an autonomous GitHub Actions runner.

## YOUR JOB:
Read the issue carefully and create an EXACT step-by-step plan to fulfill the user's request.

## CRITICAL RULES:
1. **Interpret the task LITERALLY** - do exactly what the user asks, nothing more, nothing less.
2. **Do NOT invent steps** that are not requested (e.g., do NOT clone repos unless explicitly asked).
3. **Use only available secrets** listed below. If a secret is missing, note it in the plan.
4. **For email tasks**: use SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS secrets via smtplib.
5. **For API tasks**: use the relevant API key from available secrets.
6. **Secrets are already available as environment variables** - do NOT try to read files or clone repos to get them.

## AVAILABLE SECRETS (already set as environment variables):
{secrets_info}

## ISSUE #{issue_number}: {issue_title}
{issue_body}

## RESPONSE FORMAT (strict JSON):
Return ONLY this JSON, no extra text:
{{
  "issue_number": {issue_number},
  "summary": "one sentence describing what the user wants",
  "task_type": "email|api_call|file_operation|report|other",
  "required_secrets": ["list of secrets needed from the available list above"],
  "missing_secrets": ["list of secrets needed but NOT in the available list"],
  "phases": [
    {{
      "phase": 1,
      "title": "short title",
      "description": "exact action to take",
      "tool": "python_code|shell_command|api_call",
      "expected_output": "what success looks like"
    }}
  ],
  "success_criteria": "how to verify the task succeeded"
}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2000
    )

    raw = response.choices[0].message.content.strip()

    # استخراج JSON من الرد
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    plan = json.loads(raw)

    # حفظ الخطة كـ JSON
    with open(f"task_{issue_number}.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    # حفظ الخطة كـ Markdown للقراءة
    md_lines = [
        f"# Task Plan - Issue #{issue_number}",
        f"",
        f"## Summary",
        f"{plan['summary']}",
        f"",
        f"## Task Type",
        f"{plan.get('task_type', 'other')}",
        f"",
        f"## Required Secrets",
    ]
    for s in plan.get("required_secrets", []):
        md_lines.append(f"- {s}")

    if plan.get("missing_secrets"):
        md_lines.append(f"")
        md_lines.append(f"## ⚠️ Missing Secrets")
        for s in plan["missing_secrets"]:
            md_lines.append(f"- {s}")

    md_lines += ["", "## Phases"]
    for ph in plan.get("phases", []):
        md_lines.append(f"- [ ] Phase {ph['phase']}: {ph['title']} - {ph['description']}")

    md_lines += ["", f"## Success Criteria", plan.get("success_criteria", "Task completed")]

    with open("plan_task.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"✅ Plan created for Issue #{issue_number}: {plan['summary']}")
    print(f"📋 Task type: {plan.get('task_type')}")
    print(f"🔑 Required secrets: {plan.get('required_secrets')}")
    if plan.get("missing_secrets"):
        print(f"⚠️ Missing secrets: {plan['missing_secrets']}")
    return plan


if __name__ == "__main__":
    issue_number = int(os.environ.get("ISSUE_NUMBER", "0"))
    issue_title = os.environ.get("ISSUE_TITLE", "")
    issue_body = os.environ.get("ISSUE_BODY", "")
    run_planner(issue_number, issue_title, issue_body)
