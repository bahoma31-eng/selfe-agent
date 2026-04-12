# Selfe Agent 🤖

عميل ذكاء اصطناعي بسيط يقرأ المفتاح السري من GitHub Secrets وأسماء النماذج من ملف `models.txt`.

---

## 📁 هيكل الملفات

```
selfe-agent/
├── agent.py          # السكريبت الرئيسي
├── models.txt        # قائمة أسماء النماذج
├── requirements.txt  # المكتبات المطلوبة
└── .github/
    └── workflows/
        └── run_agent.yml  # تشغيل عبر GitHub Actions
```

---

## ⚙️ الإعداد

### 1. إضافة المفتاح السري

اذهب إلى: **Settings → Secrets and variables → Actions → New repository secret**

| الاسم | القيمة |
|-------|--------|
| `OPENAI_API_KEY` | مفتاح API الخاص بك |

### 2. تثبيت المكتبات (محلياً)

```bash
pip install -r requirements.txt
```

### 3. تشغيل السكريبت محلياً

```bash
export OPENAI_API_KEY="sk-..."
python agent.py
```

---

## 🚀 التشغيل عبر GitHub Actions

1. اذهب إلى تبويب **Actions**
2. اختر **Run Selfe Agent**
3. اضغط **Run workflow**
4. أدخل رسالتك ورقم النموذج

---

## 📝 تعديل قائمة النماذج

افتح ملف `models.txt` وأضف أو احذف أسماء النماذج (سطر واحد لكل نموذج):

```
gpt-4o
gpt-4o-mini
claude-3-5-sonnet-20241022
```
