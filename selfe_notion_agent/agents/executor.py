import os
import json
import subprocess
from datetime import datetime, timezone
from groq import Groq

client = Groq(api_key=os.environ['GROQ_API_KEY'])

# ── تحويل نص المرحلة إلى أوامر حقيقية عبر LLM ────────────────────
def generate_commands(phase_text: str, issue_context: str) -> list[str]:
    prompt = f"""You are a Linux terminal command generator.
Convert this task description into a list of exact shell commands to execute.

Issue context: {issue_context}
Task: {phase_text}

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

    # استخراج JSON من الرد
    import re
    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if match:
        commands = json.loads(match.group())
        return commands
    return ['echo "Could not parse commands"']

# ── إنشاء task_N.json ────────────────────────────────────────────
def create_task_json(output_dir, issue_number, phase_index, phase_text, commands):
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

# ── تنفيذ الأوامر وتسجيل النتائج ────────────────────────────────
def execute_phase(issue_number, phase_index, phase_text, commands):
    output_dir = f"output/issue_{issue_number}"
    os.makedirs(output_dir, exist_ok=True)

    # إنشاء task_N.json
    create_task_json(output_dir, issue_number, phase_index, phase_text, commands)

    log = {
        "issue_number": issue_number,
        "phase": phase_index + 1,
        "description": phase_text,
        "commands": commands,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "stdout": "",
        "stderr": "",
        "verified": False,
        "notes": ""
    }

    all_stdout = []
    success = True

    for cmd in commands:
        print(f"  ▶ Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        all_stdout.append(f"$ {cmd}\n{result.stdout}")

        if result.returncode != 0:
            success = False
            log['stderr'] = result.stderr
            log['notes'] = f"Failed at command: {cmd}"
            break

    log['stdout'] = '\n'.join(all_stdout)
    log['finished_at'] = datetime.now(timezone.utc).isoformat()
    log['status'] = 'success' if success else 'failed'
    log['verified'] = success

    log_file = f"{output_dir}/logging_{phase_index + 1}.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(f"  {'✅' if success else '❌'} Phase {phase_index+1}: {log['status']}")
    return success

# ── قراءة plan_task.md ────────────────────────────────────────────
def parse_plan(plan_file: str):
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

# ── الدخول الرئيسي ────────────────────────────────────────────────
if __name__ == "__main__":
    issue_number = int(os.environ['ISSUE_NUMBER'])
    issue_context = os.environ.get('ISSUE_TITLE', '') + ': ' + os.environ.get('ISSUE_BODY', '')

    phases = parse_plan('plan_task.md')
    print(f"📋 Found {len(phases)} phases")

    for i, phase in enumerate(phases):
        if phase['done']:
            print(f"⏭️  Phase {i+1} already done, skipping")
            continue

        print(f"\n🔄 Phase {i+1}: {phase['text'][:60]}...")

        # تحويل النص الطبيعي → أوامر حقيقية
        commands = generate_commands(phase['text'], issue_context)
        print(f"  📦 Commands: {commands}")

        success = execute_phase(issue_number, i, phase['text'], commands)

        if success:
            mark_phase_done('plan_task.md', i)  # ← [x] فقط عند النجاح
        else:
            print(f"\n🛑 Stopped at Phase {i+1} due to failure")
            exit(1)

    print("\n🎉 All phases completed!")
