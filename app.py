import streamlit as st
import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# Set page config
st.set_page_config(page_title="AI Resume Screening Tool", layout="centered")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# -------------------- Functions ------------------------

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_cleaned_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

def get_similarity(resume_text, job_desc_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_desc_text])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100

def generate_resume_feedback(resume_text, job_desc_text, gemini_api_key):
    prompt = f"""
You are a helpful career assistant. Based on the RESUME and JOB DESCRIPTION below, give feedback on:
- Resume-job alignment
- Missing skills or keywords
- Suggestions to improve tone or content

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_desc_text}

Provide short, actionable suggestions:
"""
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating feedback: {str(e)}"

# -------------------- UI ------------------------

st.title("🤖 AI Resume Screening Tool")
st.markdown("Upload a resume PDF and paste a job description to evaluate match and get feedback.")

# Gemini API Key input
gemini_api_key = st.text_input("🔑 Enter Gemini API Key (from https://aistudio.google.com/app/apikey)", type="password")

# Upload resume and job description
resume_file = st.file_uploader("📄 Upload Resume (PDF format only)", type=["pdf"])
job_description = st.text_area("📝 Paste Job Description Here")

# Main Logic
if resume_file and job_description:
    with st.spinner("Reading and processing resume..."):
        raw_resume_text = extract_text_from_pdf(resume_file)
        clean_resume_text = get_cleaned_text(raw_resume_text)
        clean_job_desc = get_cleaned_text(job_description)

        score = get_similarity(clean_resume_text, clean_job_desc)

    st.subheader("📊 Match Result")
    st.metric(label="Match Score", value=f"{score:.2f}%")

    if score >= 75:
        st.success("✅ Excellent match! Your resume aligns well with the job.")
    elif score >= 50:
        st.warning("⚠️ Moderate match. Consider updating your resume for better alignment.")
    else:
        st.error("❌ Low match. Resume needs improvements to match this job.")

    if gemini_api_key:
        with st.spinner("🧠 Generating AI-based feedback using Gemini..."):
            feedback = generate_resume_feedback(raw_resume_text, job_description, gemini_api_key)
    else:
        feedback = "⚠️ Please enter a valid Gemini API key to get feedback."

    st.subheader("💡 Gemini AI Suggestions")
    st.info(feedback)

    st.markdown("---")
    st.subheader("📄 Extracted Resume Text")
    st.text_area("Resume Content", raw_resume_text, height=200)

