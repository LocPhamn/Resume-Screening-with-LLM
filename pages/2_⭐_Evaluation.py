import streamlit as st
import preprocess
st.set_page_config(
    page_title="Job Description & Q&A", page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
