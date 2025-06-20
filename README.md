# 🤖 AI Resume Screening Tool (Gemini-Powered)

An intelligent web app that analyzes resumes against job descriptions using **NLP and Google's Gemini AI**. It calculates a **match score** using text similarity and provides **AI-generated feedback** to help job seekers optimize their resumes for specific job roles.

---

## 🚀 Features

✅ Upload resume (PDF) and paste job description  
✅ Clean and preprocess text using spaCy  
✅ Generate match score using TF-IDF + cosine similarity  
✅ Get AI-based feedback using **Gemini (Google AI)**  
✅ Simple, clean UI built with **Streamlit**

---

## 🧰 Tech Stack

| Technology          | Purpose                                  |
|---------------------|-------------------------------------------|
| Python              | Core programming language                |
| Streamlit           | Web interface                            |
| spaCy               | NLP for text preprocessing               |
| scikit-learn        | TF-IDF vectorization + cosine similarity |
| PyPDF2              | Extract text from resume PDFs            |
| Google Generative AI (`gemini-pro`) | Generate AI feedback using Gemini API |

---

## 🧪 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/resume-screening-tool-gemini.git
cd resume-screening-tool-gemini

### ⚙️ Step 2: Run the App

Make sure you're in the project folder (where `app.py` is located), then run the following command:

```bash
python -m streamlit run app.py
