# 🤖 AI-Based Resume Screening Tool

A smart resume screening web application that uses **NLP** and **OpenAI's GPT** to analyze a resume against a job description. It provides a **match score** and **AI-generated feedback** to help job seekers optimize their resumes.

Built using:
- Python
- Streamlit
- spaCy
- OpenAI GPT-3.5
- Scikit-learn
- PyPDF2

---

## 🚀 Features

✅ Upload a resume in PDF format  
✅ Paste any job description  
✅ Calculate a **match score** using TF-IDF + cosine similarity  
✅ Generate **GPT-powered feedback** to improve your resume  
✅ View extracted resume text for transparency  

---

## 🧰 Tech Stack

| Tool        | Purpose                        |
|-------------|--------------------------------|
| Streamlit   | Frontend UI                    |
| spaCy       | NLP for tokenization & lemmatization |
| OpenAI API  | GPT-based feedback generator   |
| PyPDF2      | Resume text extraction         |
| Scikit-learn| Text similarity (TF-IDF + Cosine) |

---

## 🖥️ Demo

> 📌 _Coming soon_ – You can deploy this using [Streamlit Cloud](https://streamlit.io/cloud)

---

## 🧪 How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/resume-screening-tool.git
cd resume-screening-tool
2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
3. Run the app
streamlit run app.py

