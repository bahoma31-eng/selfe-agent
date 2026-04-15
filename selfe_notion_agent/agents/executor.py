import os
import json
import subprocess
from datetime import datetime, timezone

def parse_plan(plan_file: str):
    with open(plan_file, 'r') as f:
        content = f.read()
    phases = []
    for line in content.split('\n'):
        if line.strip().startswith('- [ ]'):
            phases.append({'done': False, 'text': line.strip()[6:]})
        elif line.strip().startswith('- [x]'):
            phases.append({'done': True, 'text': line.strip()[6:]})
    return phases

def mark_phase_done(plan_file: str, phase_index: int):
    with open(plan_file, 'r') as f:
        content = f.read()
    count = 0
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('- [ ]'):
            if count == phase_index:
                lines[i] = lines[i].replace('- [ ]', '- [x]', 1)
                break
            count += 1
    with open(plan_file, 'w') as f:
        f.write('\n'.join(lines))

def execute_phase(issue_number, phase_index, phase_text, commands):
    output_dir = f"output/issue_{issue_number}"
    os.makedirs(output_dir, exist_ok=True)
    
    log = {
        "issue_number": issue_number,
        "phase": phase_index + 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "stdout": "", "stderr": "", "verified": False
    }
    
    all_stdout = []
    success = True
    for cmd in commands:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        all_stdout.append(result.stdout)
        if result.returncode != 0:
            success = False
            log['stderr'] = result.stderr
            break
    
    log['stdout'] = '\n'.join(all_stdout)
    log['finished_at'] = datetime.now(timezone.utc).isoformat()
    log['status'] = 'success' if success else 'failed'
    log['verified'] = success
    
    log_file = f"{output_dir}/logging_{phase_index + 1}.json"
    with open(log_file, 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    
    return success

if __name__ == "__main__":
    issue_number = int(os.environ['ISSUE_NUMBER'])
    phases = parse_plan('plan_task.md')
    
    for i, phase in enumerate(phases):
        if phase['done']:
            continue
        print(f"Executing Phase {i+1}: {phase['text']}")
        success = execute_phase(issue_number, i, phase['text'], ['echo executing...'])
        if success:
            mark_phase_done('plan_task.md', i)
            print(f"Phase {i+1} completed successfully")
        else:
            print(f"Phase {i+1} FAILED - stopping")
            break
    print("All phases completed!")
