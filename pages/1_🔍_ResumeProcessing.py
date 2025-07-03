import base64
import streamlit as st
import preprocess

st.set_page_config(
    page_title="Resume Preprocess",   page_icon="👋",
)
def show_pdf(pdf_file):
    # Đọc file PDF và encode base64 để hiển thị trên Streamlit
    base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)
    pdf_file.seek(0)  # Reset lại con trỏ file để không ảnh hưởng đến các thao tác khác


st.title("Validate Resume")
pdf_files = st.file_uploader("Choose your Resume", type=["pdf"], accept_multiple_files=True)
if pdf_files:
    for i, pdf_file in enumerate(pdf_files):
        st.subheader(f"Resume {i + 1}")
        tab1, tab2, tab3 = st.tabs(["Xem PDF", "Trích xuất nội dung", "Rút gọn nội dung"])
        with tab1:
            show_pdf(pdf_file)
        with tab2:
            resume = preprocess.extract_text_from_pdf(pdf_file)
            st.write(resume)
        with tab3:
            brief_candidate = preprocess.brief_resume(pdf_file)["brief_info"]
            st.write(brief_candidate)