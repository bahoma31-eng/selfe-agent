import os
import json
import subprocess
import requests
from groq import Groq
from datetime import datetime

GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')
GITHUB_REPOSITORY = os.environ.get('GITHUB_REPOSITORY', '')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
ISSUE_NUMBER = int(os.environ.get('ISSUE_NUMBER', 0))

def get_repo_structure():
    """Scan the repository and return its structure."""
    result = {}
    ignore = {'.git', '__pycache__', 'node_modules', '.github'}
    file_list = []
    total_files = 0
    total_lines = 0
    languages = {}

    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in ignore]
        for f in files:
            path = os.path.join(root, f).lstrip('./')
            ext = os.path.splitext(f)[1]
            size = 0
            lines = 0
            try:
                full_path = os.path.join(root, f)
                size = os.path.getsize(full_path)
                if ext in ['.py', '.js', '.ts', '.yml', '.yaml', '.md', '.json', '.txt', '.sh']:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as fh:
                        lines = len(fh.readlines())
                    total_lines += lines
            except Exception:
                pass
            total_files += 1
            lang = ext if ext else 'no-ext'
            languages[lang] = languages.get(lang, 0) + 1
            file_list.append({'path': path, 'size': size, 'lines': lines, 'ext': ext})

    result['total_files'] = total_files
    result['total_lines'] = total_lines
    result['languages'] = languages
    result['files'] = file_list
    return result

def get_recent_commits():
    """Get recent git commits."""
    try:
        out = subprocess.check_output(
            ['git', 'log', '--oneline', '-10'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        return out
    except Exception:
        return 'No commits found'

def get_plan_content():
    """Read the plan_task.md file."""
    try:
        with open('plan_task.md', 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return ''

def mark_all_phases_done(plan_content):
    """Mark all phases as done in plan_task.md."""
    lines = plan_content.split('\n')
    updated = []
    for line in lines:
        if line.strip().startswith('- [ ]'):
            line = line.replace('- [ ]', '- [x]', 1)
        updated.append(line)
    result = '\n'.join(updated)
    with open('plan_task.md', 'w', encoding='utf-8') as f:
        f.write(result)
    return result

def analyze_with_groq(repo_structure, plan_content, commits):
    """Use Groq AI to analyze the repo and produce a full report."""
    client = Groq(api_key=GROQ_API_KEY)

    # Build file tree summary
    files_summary = '\n'.join([
        f"  - {f['path']} ({f['lines']} lines)" if f['lines'] > 0 else f"  - {f['path']}"
        for f in repo_structure['files'][:40]
    ])
    lang_summary = ', '.join([f"{k}: {v}" for k, v in sorted(repo_structure['languages'].items(), key=lambda x: -x[1])[:8]])

    prompt = f"""You are an expert code analyst. Analyze this GitHub repository and produce a detailed technical report in Arabic.

Repository Statistics:
- Total Files: {repo_structure['total_files']}
- Total Lines of Code: {repo_structure['total_lines']}
- Languages/Extensions: {lang_summary}

File Structure (first 40 files):
{files_summary}

Recent Commits:
{commits}

Task Plan:
{plan_content}

Write a comprehensive report in Arabic covering:
1. نظرة عامة على المستودع
2. هيكل المشروع والملفات الرئيسية
3. اللغات والتقنيات المستخدمة
4. تحليل الكود والوحدات الرئيسية
5. جودة الكود والملاحظات
6. التوصيات والاقتراحات للتحسين

Format the report nicely with emojis and markdown headers."""

    response = client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=2000
    )
    return response.choices[0].message.content

def post_comment_to_issue(comment_body):
    """Post the report as a comment on the GitHub Issue."""
    url = f'https://api.github.com/repos/{GITHUB_REPOSITORY}/issues/{ISSUE_NUMBER}/comments'
    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json'
    }
    data = {'body': comment_body}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 201:
        print(f'Comment posted successfully on Issue #{ISSUE_NUMBER}')
    else:
        print(f'Failed to post comment: {response.status_code} - {response.text}')

def save_report(report):
    """Save report to output folder."""
    os.makedirs('selfe_notion_agent/output', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'selfe_notion_agent/output/report_issue_{ISSUE_NUMBER}_{ts}.md'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'Report saved to {path}')
    return path

if __name__ == '__main__':
    print(f'[Executor] Starting execution for Issue #{ISSUE_NUMBER}...')

    # Step 1: Read plan
    plan_content = get_plan_content()
    print('[Executor] Plan loaded.')

    # Step 2: Scan repo
    print('[Executor] Scanning repository structure...')
    repo_structure = get_repo_structure()
    print(f"[Executor] Found {repo_structure['total_files']} files, {repo_structure['total_lines']} lines of code.")

    # Step 3: Get commits
    commits = get_recent_commits()
    print('[Executor] Fetched recent commits.')

    # Step 4: Analyze with Groq AI
    print('[Executor] Generating AI report with Groq...')
    report = analyze_with_groq(repo_structure, plan_content, commits)
    print('[Executor] Report generated.')

    # Step 5: Mark all phases as done
    mark_all_phases_done(plan_content)
    print('[Executor] All phases marked as done in plan_task.md.')

    # Step 6: Save report
    report_path = save_report(report)

    # Step 7: Post report as comment on Issue
    full_comment = f"""## 📊 تقرير تحليل المستودع - Issue #{ISSUE_NUMBER}

{report}

---
*تم إنشاء هذا التقرير تلقائياً بواسطة **Selfe Notion Agent** باستخدام Groq AI*
*📁 تم حفظ التقرير في: `{report_path}`*"""

    post_comment_to_issue(full_comment)
    print('[Executor] Done! Report posted on Issue.')
