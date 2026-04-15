# 🤖 Autonomous Task Agent System

نظام وكلاء مستقل يعمل على GitHub Actions لتخطيط وتنفيذ المهام تلقائياً من Issues.

## 📁 هيكل المشروع

```
selfe_notion_agent/
├── agents/
│   ├── planner.py          # Agent 1 — التخطيط
│   └── executor.py         # Agent 2 — التنفيذ
├── .github/
│   └── workflows/
│       └── agent.yml       # GitHub Actions Workflow
├── output/                 # مخرجات التنفيذ
└── README.md
```

## 🚀 طريقة الاستخدام

1. أضف Secret باسم `GROQ_API_KEY` في إعدادات الـ repo
2. افتح Issue جديد على GitHub
3. سيعمل Agent 1 تلقائياً لإنشاء خطة `plan_task.md`
4. ثم يعمل Agent 2 لتنفيذ كل مرحلة
5. تُحفظ نتائج التنفيذ في مجلد `output/`

## 🧠 الصفحة الرئيسية على Notion

[🤖 Autonomous Task Agent System](https://www.notion.so/343cf590bf7f811f9073fd117af31643)

## 📋 المتطلبات

- Python 3.11+
- `groq` library
- GROQ_API_KEY
