# Gemini-ChatBot-Streamlit-LangChain-Memory-
دردشة تفاعلية مبنية على Gemini 1.5 Flash باستخدام Streamlit وLangChain. الكود يحفظ سياق الحوار عبر st.session_state ويولّد ردود سريعة وذكية.



# Gemini ChatBot (Streamlit + LangChain + Memory)

دردشة تفاعلية تستخدم نموذج **Gemini 1.5 Flash** عبر **LangChain** وواجهة **Streamlit**.  
يحافظ التطبيق على سياق المحادثة باستخدام `st.session_state` ويعرض الردود في نفس الصفحة.

---

## ✨ المزايا
- ⚡️ سريع: يعتمد على **gemini-1.5-flash** للأداء العالي.
- 🧠 يحافظ على سياق الدردشة عبر الجلسة.
- 🧩 مبني على **LangChain** (واجهات رسائل قياسية `System/Human/AI`).
- 🖥️ واجهة سهلة عبر **Streamlit**.

---

## 📦 المتطلبات
- Python 3.10+
- حساب Google AI Studio ومفتاح API فعّال

### requirements.txt
```txt
streamlit
python-dotenv
langchain
google-generativeai
langchain-google-genai
