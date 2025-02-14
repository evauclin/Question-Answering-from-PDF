import streamlit as st
import os
import tempfile
from pathlib import Path
from query_data import query_rag
from populate_database import (
    load_documents,
    split_documents,
    add_to_chroma,
    clear_database,
)
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema.document import Document

st.set_page_config(page_title="PDF Question-Answering System", layout="wide")

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "db_initialized" not in st.session_state:
    st.session_state.db_initialized = False

st.title("📚 PDF Question-Answering System")


def process_uploaded_file(uploaded_file):
    """Process uploaded PDF file using existing functions"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded file to temp directory
        temp_path = Path(temp_dir) / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Use your existing document loader
        loader = PyPDFDirectoryLoader(temp_dir)
        documents = loader.load()

        # Process documents using your existing functions
        chunks = split_documents(documents)
        add_to_chroma(chunks)


# File upload
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file:
    if not st.session_state.db_initialized:
        with st.spinner("Processing PDF file..."):
            clear_database()  # Clear existing database
            process_uploaded_file(uploaded_file)
            st.session_state.db_initialized = True
        st.success("PDF processed successfully! You can now ask questions.")

    # Question input
    question = st.text_input("Ask a question about your document:")
    if question:
        with st.spinner("Finding answer..."):
            response = query_rag(question)
            st.session_state.conversation_history.append(
                {"question": question, "answer": response}
            )

    # Display conversation history
    if st.session_state.conversation_history:
        st.markdown("### Conversation History")
        for item in st.session_state.conversation_history:
            st.markdown(f"**Question:** {item['question']}")
            st.markdown(f"**Answer:** {item['answer']}")
            st.markdown("---")

    # Reset button
    if st.button("Reset Conversation"):
        st.session_state.conversation_history = []
        st.session_state.db_initialized = False
        clear_database()
        st.experimental_rerun()

else:
    st.info("👆 Start by uploading a PDF file.")

# Instructions in sidebar
with st.sidebar:
    st.markdown("""
    ### How to use this app:
    1. Upload a PDF file
    2. Wait for the system to process it
    3. Ask questions about the content
    4. View answers and sources
    5. Use the Reset button to start over
    """)
