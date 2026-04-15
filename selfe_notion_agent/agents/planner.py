import os
import json
from groq import Groq

def run_planner(issue_number: int, issue_title: str, issue_body: str):
    client = Groq(api_key=os.environ['GROQ_API_KEY'])
    
    prompt = f"""You are a task planning agent. Analyze the following GitHub issue and create a structured plan.

Issue #{issue_number}: {issue_title}
{issue_body}

Create a plan with maximum 10 phases. Return ONLY markdown in this format:
# Task Plan - Issue #{issue_number}

## Issue Summary
[brief summary]

## Phases
- [ ] Phase 1: [title] - [description]
- [ ] Phase 2: ..."""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    
    plan = response.choices[0].message.content
    
    with open('plan_task.md', 'w', encoding='utf-8') as f:
        f.write(plan)
    
    print(f"plan_task.md created with plan for Issue #{issue_number}")
    return plan

if __name__ == "__main__":
    issue_number = int(os.environ['ISSUE_NUMBER'])
    issue_title = os.environ['ISSUE_TITLE']
    issue_body = os.environ['ISSUE_BODY']
    run_planner(issue_number, issue_title, issue_body)
