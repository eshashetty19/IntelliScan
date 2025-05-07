import streamlit as st
import google.generativeai as genai
import os
import json
import PyPDF2 as pdf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure Gemini API
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("Please set your GOOGLE_API_KEY in the .env file.")
    st.stop()

# Function to extract text from PDF
def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Gemini response generation
def get_gemini_response(input_prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(input_prompt)
        return response.text
    except Exception as e:
        return f"❌ Error: {e}"

# Home & Deploy Buttons (Aligned in Same Row)
st.markdown(
    """
    <style>
        .home-container {
            border: 2px solid #4682B4; /* Steel Blue border */
            display: inline-block;
            padding: 5px 15px;
            border-radius: 6px;
            transition: background-color 0.3s;
        }
        .home-container:hover {
            background-color: #E0F2FE; /* Light cyan on hover */
        }
        .home-link {
            color: black;
            font-weight: bold;
            text-decoration: none;
        }
    </style>
    <div style='text-align: left;'>
        <div class="home-container">
            <a href="https://home-page-4mkk.onrender.com" class="home-link">Home</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title("Smart ATS 🤖")
st.write("### Improve Your Resume with AI-powered ATS Matching")

# Input fields
jd = st.text_area("📄 Paste the Job Description here")
uploaded_file = st.file_uploader("📎 Upload Your Resume (PDF)", type="pdf")

# Submit button
if st.button("🔍 Analyze Resume"):
    if uploaded_file is None or jd.strip() == "":
        st.warning("Please upload a resume and paste a job description.")
    else:
        with st.spinner("Analyzing..."):
            resume_text = input_pdf_text(uploaded_file)

            # Updated prompt for front-end developer JD
            input_prompt = f"""
Hey, act like a skilled and experienced ATS (Applicant Tracking System) with deep knowledge of roles such as front-end developer, web developer, or UI/UX developer.

Your task is to evaluate the candidate's resume against the provided job description. The job market is very competitive, so provide detailed and helpful feedback.

1. Assign a *match percentage* between resume and JD, based on basic front-end skills (HTML, CSS, JavaScript).
2. List *missing keywords* from the JD based on the specific technologies required for front-end development.
3. Write a *brief profile summary* based on the resume.

Resume:
{resume_text}

Job Description:
{jd}

Respond only in JSON format like:
{{
  "JD Match": "85%",
  "MissingKeywords": ["JavaScript", "React", "Version Control"],
  "ProfileSummary": "The candidate has strong experience in front-end development using HTML, CSS, and JavaScript, but lacks experience with modern JavaScript frameworks like React, Angular, or version control tools."
}}
"""
            gemini_response = get_gemini_response(input_prompt)

        # Show result
        st.subheader("📊 ATS Analysis Result")

        try:
            # Clean up potential non-JSON text
            cleaned_response = gemini_response.strip().split("json")[-1].split("```")[0].strip()

            # Attempt to parse the JSON
            result = json.loads(cleaned_response)

            # Extract and display match percentage
            match_str = result.get("JD Match", "0").replace('%', '')
            match_percent = int(match_str)

            # 🔵 Show score as progress bar
            st.markdown(f"*✅ JD Match:* {result.get('JD Match', 'N/A')}")
            st.progress(match_percent)

            st.markdown("*⚠️ Missing Keywords:*")
            st.write(", ".join(result.get("MissingKeywords", [])))
            st.markdown("*📝 Profile Summary:*")
            st.write(result.get("ProfileSummary", "Not available"))

            # ✅ ✍️ Recommendations section
            if result.get("MissingKeywords"):
                st.markdown("### ✍️ Recommendations")
                for kw in result["MissingKeywords"]:
                    st.write(f"• Consider adding experience or projects involving *{kw}*.")

        except Exception as e:
            st.error(f"❌ Failed to parse response: {e}")
            st.markdown("### 🔎 Gemini Raw Output:")
            st.code(gemini_response, language="text")
