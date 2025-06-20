import streamlit as st
import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# ----------------- CONFIGURATION ------------------
st.set_page_config(page_title="AI Resume Matcher", layout="centered")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# ----------------- FUNCTIONS -----------------------

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

def generate_resume_feedback(resume_text, job_desc_text):
    prompt = f"""
You are a helpful career assistant. Given the following RESUME and JOB DESCRIPTION, evaluate how well the resume matches the job. Provide feedback on:
- Missing skills or keywords
- Tone/professionalism
- Suggestions to improve alignment with the job

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_desc_text}

Provide a short, actionable feedback paragraph:
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error generating feedback: {str(e)}"

# ----------------- UI -----------------------------

st.title("🤖 AI-Based Resume Screening Tool")
st.markdown("Upload a resume and paste a job description to see how well they match!")

# Optional API Key input
api_key_input = st.text_input("🔑 Enter OpenAI API Key", type="password")
if api_key_input:
    openai.api_key = api_key_input
else:
    st.warning("Please enter your OpenAI API key to enable GPT-based suggestions.")

# Upload resume and job description
resume_file = st.file_uploader("📄 Upload Resume (PDF format only)", type=["pdf"])
job_description = st.text_area("📝 Paste Job Description Here")

# Main processing
if resume_file and job_description and api_key_input:
    with st.spinner("Reading resume..."):
        raw_resume_text = extract_text_from_pdf(resume_file)
        clean_resume_text = get_cleaned_text(raw_resume_text)
        clean_job_desc = get_cleaned_text(job_description)

        score = get_similarity(clean_resume_text, clean_job_desc)

    st.subheader("📊 Match Result")
    st.metric(label="Match Score", value=f"{score:.2f}%")

    if score >= 75:
        st.success("✅ Excellent match! Ready to apply.")
    elif score >= 50:
        st.warning("⚠️ Moderate match. Consider tweaking your resume.")
    else:
        st.error("❌ Low match. Update your resume to better align with the job description.")

    with st.spinner("🧠 Generating AI-based feedback..."):
        feedback = generate_resume_feedback(raw_resume_text, job_description)

    st.subheader("💡 GPT-Based Suggestions")
    st.info(feedback)

    st.markdown("---")
    st.subheader("📄 Extracted Resume Text")
    st.text_area("Resume Content", raw_resume_text, height=200)
