import base64
import os

import streamlit as st
import preprocess

def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


st.set_page_config(
    page_title="Job Description & Q&A", page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)
PDF_path = r"D:\Python plus\NLP_AI\src\nlp\proj\resume_evaluation-upgrade\resume"

st.title("Job Description & Q&A")
jd_path = st.file_uploader("Tải lên Job Description (JD) của bạn", type=["txt", "pdf"])
if jd_path:
    brief_jd = preprocess.jd_requirement(jd_path)
    st.write("**Nội dung JD đã rút gọn:**")
    st.write(brief_jd["response"])
    top_candidates = preprocess.evaluate_resume(brief_jd["response"])
    st.title("🎖️ Top Candidates")
    st.write("Dưới đây là những ứng viên phù hợp nhất với JD của bạn:")
    st.write(top_candidates['response'])
    st.title("PDF của ứng viên")
    for i, candidate in enumerate(top_candidates['candidates_info']):
        pdf_path = os.path.join(PDF_path, candidate['name']+".pdf")
        print(pdf_path)
        show_pdf(pdf_path)

