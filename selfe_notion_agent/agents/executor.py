import os
import json
import re
import subprocess
from datetime import datetime, timezone
from groq import Groq

client = Groq(api_key=os.environ['GROQ_API_KEY'])

MAX_RETRIES = 3  # أقصى عدد لإعادة المحاولة لكل مرحلة

# ══════════════════════════════════════════════════════════════════
# 1. توليد الأوامر عبر LLM
# ══════════════════════════════════════════════════════════════════
def generate_commands(phase_text: str, issue_context: str, observation: str = "") -> list[str]:
    """
    Act — يحوّل نص المرحلة إلى أوامر shell حقيقية.
    إذا وُجدت ملاحظة (observation) فهي نتيجة محاولة سابقة فاشلة.
    """
    retry_block = ""
    if observation:
        retry_block = f"""
Previous attempt FAILED. Here is the observation:
{observation}

Think carefully about what went wrong and generate FIXED commands.
"""

    prompt = f"""You are a Linux terminal command generator.
Convert this task description into a list of exact shell commands to execute.

Issue context: {issue_context}
Task: {phase_text}
{retry_block}
Rules:
- Return ONLY a JSON array of shell command strings, nothing else.
- Commands must be safe and executable in Ubuntu GitHub Actions environment.
- Use pip, python, git, curl, mkdir, echo, etc.
- If task is unclear or conceptual only, return ["echo 'Phase noted - no commands needed'"]

Example output:
["pip install requests", "python script.py", "echo done"]
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    raw = response.choices[0].message.content.strip()
    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return ['echo "Could not parse commands"']


# ══════════════════════════════════════════════════════════════════
# 2. إنشاء task_N.json
# ══════════════════════════════════════════════════════════════════
def create_task_json(output_dir: str, issue_number: int, phase_index: int,
                     phase_text: str, commands: list) -> str:
    task = {
        "issue_number": issue_number,
        "phase": phase_index + 1,
        "title": phase_text.split("—")[0].strip() if "—" in phase_text else phase_text[:50],
        "description": phase_text,
        "commands": commands,
        "expected_output": "successful execution with returncode 0",
        "status": "pending"
    }
    task_file = f"{output_dir}/task_{phase_index + 1}.json"
    with open(task_file, 'w', encoding='utf-8') as f:
        json.dump(task, f, indent=2, ensure_ascii=False)
    return task_file


# ══════════════════════════════════════════════════════════════════
# 3. OBSERVE — قراءة task_N.json والتحقق من النجاح
# ══════════════════════════════════════════════════════════════════
def observe_task(output_dir: str, phase_index: int) -> dict:
    """
    يقرأ task_N.json ويُرجع بيانات المهمة الحالية.
    """
    task_file = f"{output_dir}/task_{phase_index + 1}.json"
    if not os.path.exists(task_file):
        return {}
    with open(task_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def update_task_status(output_dir: str, phase_index: int, status: str, notes: str = ""):
    """
    يحدّث حقل status في task_N.json بعد التنفيذ.
    """
    task_file = f"{output_dir}/task_{phase_index + 1}.json"
    if not os.path.exists(task_file):
        return
    with open(task_file, 'r', encoding='utf-8') as f:
        task = json.load(f)
    task['status'] = status
    if notes:
        task['notes'] = notes
    with open(task_file, 'w', encoding='utf-8') as f:
        json.dump(task, f, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════
# 4. THINK — تحليل الفشل وتوليد ملاحظة للمحاولة التالية
# ══════════════════════════════════════════════════════════════════
def think_about_failure(phase_text: str, commands: list, stderr: str, stdout: str, attempt: int) -> str:
    """
    يُرسل الخطأ إلى LLM ليفكر في السبب ويُنتج ملاحظة مفيدة للمحاولة التالية.
    """
    prompt = f"""You are a debugging AI agent. A shell command failed. Analyze and explain what went wrong.

Task: {phase_text}
Attempt number: {attempt}
Commands tried: {json.dumps(commands)}
STDOUT: {stdout[:1000]}
STDERR: {stderr[:1000]}

Provide a short, precise observation (2-4 sentences) about:
1. What went wrong
2. What should be changed in the next attempt

Be specific and actionable.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════════
# 5. ACT — تنفيذ الأوامر وتسجيل النتائج
# ══════════════════════════════════════════════════════════════════
def run_commands(commands: list) -> dict:
    """
    ينفّذ قائمة الأوامر ويُرجع نتيجة التنفيذ.
    """
    all_stdout = []
    last_stderr = ""
    success = True
    failed_cmd = ""

    for cmd in commands:
        print(f"  ▶ Running: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
            all_stdout.append(f"$ {cmd}\n{result.stdout}")
            if result.returncode != 0:
                success = False
                last_stderr = result.stderr
                failed_cmd = cmd
                break
        except subprocess.TimeoutExpired:
            success = False
            last_stderr = f"Command timed out after 120s: {cmd}"
            failed_cmd = cmd
            break

    return {
        "success": success,
        "stdout": "\n".join(all_stdout),
        "stderr": last_stderr,
        "failed_cmd": failed_cmd
    }


# ══════════════════════════════════════════════════════════════════
# 6. حلقة Observe → Think → Act الرئيسية لكل مرحلة
# ══════════════════════════════════════════════════════════════════
def execute_phase_with_retry(issue_number: int, phase_index: int,
                              phase_text: str, issue_context: str) -> bool:
    output_dir = f"output/issue_{issue_number}"
    os.makedirs(output_dir, exist_ok=True)

    observation = ""  # فارغة في أول محاولة

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n  🔁 Attempt {attempt}/{MAX_RETRIES} — Phase {phase_index + 1}")

        # ── ACT: توليد الأوامر (مع الملاحظة إن وُجدت) ────────────
        commands = generate_commands(phase_text, issue_context, observation)
        print(f"  📦 Commands: {commands}")

        # حفظ task_N.json (يُحدَّث في كل محاولة)
        create_task_json(output_dir, issue_number, phase_index, phase_text, commands)

        # ── ACT: تنفيذ الأوامر ────────────────────────────────────
        run_result = run_commands(commands)

        # ── OBSERVE: قراءة task_N.json والتحقق ───────────────────
        task_data = observe_task(output_dir, phase_index)
        print(f"  🔍 Observing task_{phase_index + 1}.json — status: {task_data.get('status', 'pending')}")

        if run_result["success"]:
            # ✅ نجاح — تحديث الحالة وحفظ السجل
            update_task_status(output_dir, phase_index, "success")
            _save_log(output_dir, phase_index, phase_text, commands,
                      run_result["stdout"], "", attempt, True)
            print(f"  ✅ Phase {phase_index + 1} succeeded on attempt {attempt}")
            return True
        else:
            # ❌ فشل — THINK: تحليل الخطأ وتوليد ملاحظة
            print(f"  ❌ Attempt {attempt} failed: {run_result['stderr'][:200]}")
            update_task_status(output_dir, phase_index, f"failed_attempt_{attempt}",
                               f"Failed at: {run_result['failed_cmd']}")

            if attempt < MAX_RETRIES:
                # THINK
                print("  🧠 Thinking about what went wrong...")
                observation = think_about_failure(
                    phase_text, commands,
                    run_result["stderr"], run_result["stdout"], attempt
                )
                print(f"  💡 Observation: {observation[:200]}")
            else:
                # استنفذنا كل المحاولات
                _save_log(output_dir, phase_index, phase_text, commands,
                          run_result["stdout"], run_result["stderr"], attempt, False)
                update_task_status(output_dir, phase_index, "failed_all_retries",
                                   f"Failed after {MAX_RETRIES} attempts. Last error: {run_result['stderr'][:300]}")
                print(f"  🛑 Phase {phase_index + 1} failed after {MAX_RETRIES} attempts")
                return False

    return False


def _save_log(output_dir, phase_index, phase_text, commands, stdout, stderr, attempt, success):
    log = {
        "phase": phase_index + 1,
        "description": phase_text,
        "commands": commands,
        "attempt": attempt,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "status": "success" if success else "failed",
        "verified": success,
        "stdout": stdout,
        "stderr": stderr
    }
    log_file = f"{output_dir}/logging_{phase_index + 1}.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════
# 7. قراءة plan_task.md وتحديثه
# ══════════════════════════════════════════════════════════════════
def parse_plan(plan_file: str) -> list:
    with open(plan_file, 'r', encoding='utf-8') as f:
        content = f.read()
    phases = []
    for line in content.split('\n'):
        if line.strip().startswith('- [ ]'):
            phases.append({'done': False, 'text': line.strip()[6:]})
        elif line.strip().startswith('- [x]'):
            phases.append({'done': True, 'text': line.strip()[6:]})
    return phases


def mark_phase_done(plan_file: str, phase_index: int):
    with open(plan_file, 'r', encoding='utf-8') as f:
        content = f.read()
    count = 0
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('- [ ]'):
            if count == phase_index:
                lines[i] = lines[i].replace('- [ ]', '- [x]', 1)
                break
            count += 1
    with open(plan_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# ══════════════════════════════════════════════════════════════════
# 8. الدخول الرئيسي
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    issue_number = int(os.environ['ISSUE_NUMBER'])
    issue_context = (os.environ.get('ISSUE_TITLE', '') + ': '
                     + os.environ.get('ISSUE_BODY', ''))

    phases = parse_plan('plan_task.md')
    print(f"📋 Found {len(phases)} phases")

    for i, phase in enumerate(phases):
        if phase['done']:
            print(f"⏭️  Phase {i+1} already done, skipping")
            continue

        print(f"\n🔄 Phase {i+1}: {phase['text'][:70]}...")

        # ── Observe → Think → Act loop ─────────────────────────
        success = execute_phase_with_retry(issue_number, i, phase['text'], issue_context)

        if success:
            mark_phase_done('plan_task.md', i)
        else:
            print(f"\n🛑 Stopped at Phase {i+1} after all retries failed.")
            exit(1)

    print("\n🎉 All phases completed successfully!")
