import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(sender_email, sender_password, receiver_email, subject, body):
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        smtp_server = "smtp.gmail.com"
        smtp_port = 587

        print(f"جاري الاتصال بخادم SMTP: {smtp_server}:{smtp_port}...")
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        print("تم تفعيل التشفير TLS.")

        server.login(sender_email, sender_password)
        print("تم تسجيل الدخول بنجاح.")

        server.sendmail(sender_email, receiver_email, msg.as_string())
        print("✅ تم إرسال البريد الإلكتروني بنجاح!")

    except smtplib.SMTPAuthenticationError:
        print("❌ خطأ في المصادقة: تحقق من SMTP_USER و SMTP_PASS.")
        raise
    except Exception as e:
        print(f"❌ خطأ: {e}")
        raise
    finally:
        try:
            server.quit()
        except Exception:
            pass

if __name__ == "__main__":
    SENDER_EMAIL = os.getenv("SMTP_USER")
    SENDER_PASSWORD = os.getenv("SMTP_PASS")

    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("❌ خطأ: متغيرات البيئة SMTP_USER أو SMTP_PASS غير موجودة.")
        exit(1)

    RECEIVER_EMAIL = "bahoma31@gmail.com"
    EMAIL_SUBJECT = "رسالة تلقائية من Selfe Agent 🤖"
    EMAIL_BODY = """مرحباً،

هذه رسالة تلقائية تم إرسالها عبر GitHub Actions بواسطة Selfe Agent.

مع التحيات،
Selfe Agent"""

    send_email(SENDER_EMAIL, SENDER_PASSWORD, RECEIVER_EMAIL, EMAIL_SUBJECT, EMAIL_BODY)
