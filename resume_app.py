import streamlit as st
import tempfile
import os
from resume_tool import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_key_points,
    generate_bullet_points,
    calculate_keyword_match,
    create_new_resume
)

# Configure page
st.set_page_config(
    page_title="AI Resume Rewriter",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 AI Resume Rewriter")
st.caption("Upload your resume and job description to get AI-powered optimization suggestions")

with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_resume = st.file_uploader("Resume (.pdf, .docx)", type=["pdf", "docx"], help="Upload your current resume")
    
    st.subheader("Job Description")
    job_input_method = st.radio("Input method:", ["Paste Text", "Upload File"], horizontal=True)
    
    job_description = None
    if job_input_method == "Paste Text":
        job_description = st.text_area("Paste job description:", height=200)
    else:
        uploaded_job = st.file_uploader("Job Description (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])
        if uploaded_job:
            with st.spinner("Extracting text..."):
                if uploaded_job.name.endswith(".pdf"):
                    job_description = extract_text_from_pdf(uploaded_job)
                elif uploaded_job.name.endswith(".docx"):
                    job_description = extract_text_from_docx(uploaded_job)
                else:  # TXT file
                    job_description = uploaded_job.read().decode("utf-8")

    st.markdown("---")
    st.caption("🛠️ This tool will:")
    st.caption("- 🔍 Analyze resume/job match")
    st.caption("- ✨ Suggest bullet point improvements")
    st.caption("- 📄 Generate optimized resume")

# Main processing
if uploaded_resume is not None and job_description:
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Resume Analysis")
        with st.spinner("Processing resume..."):
            if uploaded_resume.name.endswith(".pdf"):
                resume_text = extract_text_from_pdf(uploaded_resume)
            else:  # DOCX
                resume_text = extract_text_from_docx(uploaded_resume)
        
        # Calculate match score
        with st.spinner("Calculating match..."):
            score = calculate_keyword_match(resume_text, job_description)
        
        st.metric("🔍 Keyword Match Score", f"{score}%", 
                 help="Percentage of keywords matching between resume and job description")
        
        # Extract key points
        with st.expander("View Extracted Resume Content"):
            with st.spinner("Identifying key points..."):
                key_points = extract_key_points(resume_text)
            st.write(key_points)

    with col2:
        st.subheader("Optimization Suggestions")
        with st.spinner("Generating AI suggestions..."):
            bullets = generate_bullet_points(resume_text, key_points, job_description)
        
        # Display suggestions
        with st.expander("✨ Suggested Resume Improvements", expanded=True):
            for i, b in enumerate(bullets, 1):
                st.markdown(f"{i}. {b}")
        
        # Download section - FIXED FILE HANDLING
        st.subheader("Download Enhanced Resume")
        if st.button("🔄 Generate Updated Resume", type="primary", use_container_width=True):
            with st.spinner("Creating document..."):
                # Create a temporary physical file for processing
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                    # Write uploaded content to physical file
                    tmp.write(uploaded_resume.getvalue())
                    tmp_path = tmp.name
                
                try:
                    # Create output file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as output_tmp:
                        output_path = output_tmp.name
                    
                    # Generate new resume using physical file paths
                    create_new_resume(tmp_path, bullets, output_path)
                    
                    # Read generated content
                    with open(output_path, "rb") as f:
                        output_data = f.read()
                    
                    # Determine filename
                    if "." in uploaded_resume.name:
                        base_name = uploaded_resume.name.rsplit(".", 1)[0]
                        new_name = f"{base_name}_optimized.docx"
                    else:
                        new_name = "optimized_resume.docx"
                    
                    # Create download button
                    st.download_button(
                        label="📥 Download Enhanced Resume",
                        data=output_data,
                        file_name=new_name,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True
                    )
                    st.success("Resume generated successfully!")
                
                except Exception as e:
                    st.error(f"Error generating resume: {str(e)}")
                
                finally:
                    # Clean up temporary files
                    os.unlink(tmp_path)
                    os.unlink(output_path)

elif not uploaded_resume and job_description:
    st.warning("⚠️ Please upload your resume")
elif uploaded_resume and not job_description:
    st.warning("⚠️ Please provide job description")
else:
    # Initial state instructions
    st.info("👋 Welcome! Please upload your resume and job description to get started")
    st.image("https://via.placeholder.com/800x400?text=Upload+Documents+to+Begin", use_column_width=True)