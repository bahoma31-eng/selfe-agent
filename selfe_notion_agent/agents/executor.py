import os
import json
import re
import subprocess
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone
from groq import Groq

client = Groq(api_key=os.environ['GROQ_API_KEY'])
MAX_RETRIES = 3


# ─────────────────────────────────────────────
# تحميل الـ secrets من ملف .env أو من بيئة العمل
# ─────────────────────────────────────────────
def load_secrets():
    """تحميل كل الـ secrets المتاحة كـ dict"""
    secrets = {}
    env_file = os.environ.get('SECRETS_ENV_FILE', 'agent_secrets.env')
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, _, val = line.partition('=')
                    key = key.strip()
                    val = val.strip()
                    if val and val != '${}':
                        secrets[key] = val
                        os.environ[key] = val  # تعيين في بيئة العمل أيضاً
    # إضافة ما هو موجود مباشرة في بيئة العمل
    for k in ['GROQ_API_KEY', 'GITHUB_TOKEN', 'SMTP_HOST', 'SMTP_PORT',
              'SMTP_USER', 'SMTP_PASS', 'NOTION_TOKEN', 'OPENAI_API_KEY', 'PAT_TOKEN']:
        if os.environ.get(k):
            secrets[k] = os.environ[k]
    return secrets


# ─────────────────────────────────────────────
# قراءة خطة المهمة
# ─────────────────────────────────────────────
def load_task_plan(issue_number: int) -> dict:
    plan_file = f"task_{issue_number}.json"
    if os.path.exists(plan_file):
        with open(plan_file, encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_task_result(issue_number: int, result: dict):
    plan_file = f"task_{issue_number}.json"
    plan = load_task_plan(issue_number)
    plan['execution_result'] = result
    plan['executed_at'] = datetime.now(timezone.utc).isoformat()
    with open(plan_file, 'w', encoding='utf-8') as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────
# 1. توليد كود Python عبر LLM
# ─────────────────────────────────────────────
def generate_python_code(phase: dict, issue_context: dict, secrets: dict, observation: str = "") -> str:
    available_secrets = list(secrets.keys())
    retry_note = f"\n\nPREVIOUS ATTEMPT FAILED:\n{observation}\nFix the issue in your new code." if observation else ""

    prompt = f"""You are an executor agent. Write COMPLETE, RUNNABLE Python code to accomplish this task phase.

## TASK CONTEXT:
Issue #{issue_context.get('issue_number')}: {issue_context.get('issue_title')}
{issue_context.get('issue_body')}

## CURRENT PHASE:
Phase {phase.get('phase')}: {phase.get('title')}
Description: {phase.get('description')}
Expected output: {phase.get('expected_output')}
Task type: {issue_context.get('task_type', 'other')}

## AVAILABLE ENVIRONMENT VARIABLES (already set, use os.environ.get()):
{json.dumps(available_secrets, indent=2)}

## STRICT RULES:
1. Write ONLY executable Python code, no explanations.
2. Use os.environ.get('SECRET_NAME') to access secrets - they are already set.
3. For email: use smtplib with SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS.
4. Do NOT clone git repos. Do NOT create fake credentials.
5. Print a clear SUCCESS or FAILURE message at the end.
6. Handle exceptions and print error details.
7. The code must be self-contained (import everything it needs).{retry_note}

Write the Python code now:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2000
    )
    raw = response.choices[0].message.content.strip()
    # استخراج الكود
    if "```python" in raw:
        raw = raw.split("```python")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    return raw


# ─────────────────────────────────────────────
# 2. تنفيذ الكود وجمع النتائج (Observe)
# ─────────────────────────────────────────────
def execute_code(code: str, secrets: dict) -> tuple[bool, str]:
    """تنفيذ كود Python وإرجاع (نجح؟, مخرجات)"""
    env = os.environ.copy()
    env.update(secrets)

    tmp_file = "/tmp/agent_task.py"
    with open(tmp_file, 'w', encoding='utf-8') as f:
        f.write(code)

    try:
        result = subprocess.run(
            ["python3", tmp_file],
            capture_output=True, text=True, timeout=60, env=env
        )
        output = result.stdout + result.stderr
        success = result.returncode == 0 and "SUCCESS" in result.stdout.upper()
        return success, output
    except subprocess.TimeoutExpired:
        return False, "ERROR: Execution timed out after 60 seconds"
    except Exception as e:
        return False, f"ERROR: {str(e)}"


# ─────────────────────────────────────────────
# 3. حلقة Observe → Think → Act لكل مرحلة
# ─────────────────────────────────────────────
def run_phase_with_retry(phase: dict, issue_context: dict, secrets: dict) -> dict:
    observation = ""
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n{'='*50}")
        print(f"Phase {phase['phase']}: {phase['title']} | Attempt {attempt}/{MAX_RETRIES}")

        # THINK: توليد الكود
        print("🤔 Thinking: generating code...")
        code = generate_python_code(phase, issue_context, secrets, observation)
        print(f"📝 Generated {len(code.splitlines())} lines of code")

        # ACT: تنفيذ الكود
        print("⚡ Acting: executing code...")
        success, output = execute_code(code, secrets)

        # OBSERVE: مراقبة النتيجة
        print(f"👁 Observation: {'SUCCESS' if success else 'FAILURE'}")
        print(f"Output: {output[:500]}")

        if success:
            return {
                "phase": phase['phase'],
                "title": phase['title'],
                "status": "success",
                "attempts": attempt,
                "output": output[:1000]
            }

        # تحضير ملاحظة للمحاولة التالية
        observation = f"Attempt {attempt} output:\n{output[:800]}"

    return {
        "phase": phase['phase'],
        "title": phase['title'],
        "status": "failed",
        "attempts": MAX_RETRIES,
        "output": observation[:1000]
    }


# ─────────────────────────────────────────────
# 4. نشر تعليق على الـ Issue
# ─────────────────────────────────────────────
def post_issue_comment(issue_number: int, body: str, github_token: str, repo: str):
    try:
        import urllib.request
        url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments"
        data = json.dumps({"body": body}).encode()
        req = urllib.request.Request(url, data=data, method='POST')
        req.add_header("Authorization", f"token {github_token}")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req) as resp:
            print(f"✅ Comment posted: {resp.status}")
    except Exception as e:
        print(f"⚠️ Could not post comment: {e}")


# ─────────────────────────────────────────────
# 5. المُنسّق الرئيسي
# ─────────────────────────────────────────────
def run_executor():
    issue_number = int(os.environ.get("ISSUE_NUMBER", "0"))
    issue_title = os.environ.get("ISSUE_TITLE", "")
    issue_body = os.environ.get("ISSUE_BODY", "")
    github_token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")

    print(f"🚀 Executor starting for Issue #{issue_number}: {issue_title}")

    # تحميل الـ secrets
    secrets = load_secrets()
    print(f"🔑 Loaded {len(secrets)} secrets: {list(secrets.keys())}")

    # تحميل الخطة من planner
    plan = load_task_plan(issue_number)
    if not plan:
        print("⚠️ No task plan found! Cannot proceed.")
        return

    print(f"📋 Task type: {plan.get('task_type')}")
    print(f"📌 Summary: {plan.get('summary')}")

    # التحقق من الـ secrets المفقودة
    missing = plan.get("missing_secrets", [])
    if missing:
        comment = f"⚠️ **Cannot complete task** - Missing secrets: `{'`, `'.join(missing)}`\n\nPlease add these secrets in repository Settings → Secrets and variables → Actions."
        if github_token and repo:
            post_issue_comment(issue_number, comment, github_token, repo)
        print(f"❌ Missing secrets: {missing}")
        return

    issue_context = {
        "issue_number": issue_number,
        "issue_title": issue_title,
        "issue_body": issue_body,
        "task_type": plan.get("task_type", "other"),
        "summary": plan.get("summary", "")
    }

    # تنفيذ كل مرحلة
    phases = plan.get("phases", [])
    results = []
    all_success = True

    for phase in phases:
        result = run_phase_with_retry(phase, issue_context, secrets)
        results.append(result)
        if result["status"] == "failed":
            all_success = False
            print(f"❌ Phase {result['phase']} failed after {result['attempts']} attempts")

    # حفظ النتائج
    execution_summary = {
        "status": "success" if all_success else "partial_failure",
        "phases_results": results,
        "completed_at": datetime.now(timezone.utc).isoformat()
    }
    save_task_result(issue_number, execution_summary)

    # تحديث plan_task.md
    with open("plan_task.md", "a", encoding="utf-8") as f:
        f.write(f"\n\n## Execution Results\n")
        f.write(f"**Status**: {'✅ All phases completed' if all_success else '⚠️ Some phases failed'}\n\n")
        for r in results:
            icon = "✅" if r["status"] == "success" else "❌"
            f.write(f"- {icon} Phase {r['phase']}: {r['title']} ({r['attempts']} attempt(s))\n")

    # نشر تعليق على الـ Issue
    if github_token and repo:
        lines = [f"## Agent Execution Report - Issue #{issue_number}\n"]
        lines.append(f"**Task**: {plan.get('summary')}\n")
        lines.append(f"**Status**: {'✅ Completed successfully' if all_success else '⚠️ Completed with issues'}\n\n")
        lines.append("### Phase Results\n")
        for r in results:
            icon = "✅" if r["status"] == "success" else "❌"
            lines.append(f"{icon} **Phase {r['phase']}**: {r['title']} - {r['status']} ({r['attempts']} attempt(s))\n")
        comment_body = "".join(lines)
        post_issue_comment(issue_number, comment_body, github_token, repo)

    print(f"\n{'='*50}")
    print(f"✅ Executor finished. Status: {execution_summary['status']}")


if __name__ == "__main__":
    run_executor()
