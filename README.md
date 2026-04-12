# Selfe Agent 🤖 v2.0

عميل ذكاء اصطناعي يدعم **مفاتيح متعددة** مع تدوير تلقائي عند استنفاد الحصة.

---

## 📁 هيكل الملفات

```
selfe-agent/
├── agent.py                    # السكريبت الرئيسي
├── models.txt                  # قائمة النماذج + المزود
├── requirements.txt            # المكتبات المطلوبة
└── .github/
    └── workflows/
        └── run_agent.yml       # تشغيل عبر GitHub Actions
```

---

## 🔑 المفاتيح السرية المطلوبة

اذهب إلى: **Settings → Secrets and variables → Actions → New repository secret**

| المتغير السري | المزود | الوصف |
|---|---|---|
| `GEMINI_API_KEY_1` | Google Gemini | المفتاح الأول |
| `GEMINI_API_KEY_2` | Google Gemini | المفتاح الثاني |
| `GEMINI_API_KEY_3` | Google Gemini | المفتاح الثالث |
| `GEMINI_API_KEY_4` | Google Gemini | المفتاح الرابع |
| `GROQ_API_KEY`     | Groq          | مفتاح Groq |

> عند وصول أحد مفاتيح Gemini لحد الاستخدام، يتحول السكريبت تلقائياً للمفتاح التالي.

---

## ⚙️ تشغيل محلي

```bash
pip install -r requirements.txt

export GEMINI_API_KEY_1="AIza..."
export GEMINI_API_KEY_2="AIza..."
export GEMINI_API_KEY_3="AIza..."
export GEMINI_API_KEY_4="AIza..."
export GROQ_API_KEY="gsk_..."

python agent.py
```

---

## 📝 تعديل قائمة النماذج

افتح `models.txt` وأضف نماذج بهذه الصيغة:

```
aسم_النموذج | اسم_المزود
```

مثال:
```
gemini-2.0-flash       | gemini
llama-3.3-70b-versatile | groq
```

المزودون المدعومون: `gemini` و `groq`

---

## 🚀 تشغيل عبر GitHub Actions

1. اذهب إلى تبويب **Actions**
2. اختر **Run Selfe Agent**
3. اضغط **Run workflow**
4. أدخل رسالتك ورقم النموذج المطلوب
