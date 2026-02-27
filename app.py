import streamlit as st
import os
import tempfile
from pathlib import Path
import logging

from query_data import query_rag  # Ensure that query_data.py defines query_rag
from populate_database import load_pdf_documents, split_pdf_documents, update_chroma_database, clear_database
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document

# Configure logging for debugging purposes
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

st.set_page_config(page_title="PDF Question-Answering System", layout="wide")

# Initialize session state variables
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "database_initialized" not in st.session_state:
    st.session_state.database_initialized = False

st.title("📚 PDF Question-Answering System")


def process_pdf_upload(pdf_file) -> None:
    """
    Processes the uploaded PDF file: saves it temporarily, loads its documents,
    splits the content into chunks, and adds them to the vector database.

    :param pdf_file: The PDF file uploaded by the user.
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the uploaded PDF to a temporary directory
            temp_pdf_path = Path(temp_dir) / pdf_file.name
            with open(temp_pdf_path, "wb") as temp_file:
                temp_file.write(pdf_file.getvalue())

            # Load PDF documents from the temporary directory
            pdf_loader = PyPDFDirectoryLoader(temp_dir)
            pdf_documents = pdf_loader.load()

            # Split documents into chunks and update the vector database
            document_chunks = split_pdf_documents(pdf_documents)
            update_chroma_database(document_chunks)
            logging.info("PDF file processed and added to the database.")
    except Exception as error:
        logging.error("Error processing PDF file: %s", error)
        st.error("An error occurred while processing the PDF file.")


# File uploader
uploaded_pdf = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_pdf:
    if not st.session_state.database_initialized:
        with st.spinner("Processing PDF file..."):
            clear_database()  # Reset the existing database
            process_pdf_upload(uploaded_pdf)
            st.session_state.database_initialized = True
        st.success("PDF processed successfully! You can now ask questions.")

    # User question input
    user_question = st.text_input("Ask a question about your document:")
    if user_question:
        with st.spinner("Finding answer..."):
            try:
                answer = query_rag(user_question)
            except Exception as error:
                logging.error("Error fetching answer: %s", error)
                answer = "An error occurred while fetching the answer."
            st.session_state.conversation_history.append({
                "question": user_question,
                "answer": answer
            })

    # Display conversation history
    if st.session_state.conversation_history:
        st.markdown("### Conversation History")
        for entry in st.session_state.conversation_history:
            st.markdown(f"**Question:** {entry['question']}")
            st.markdown(f"**Answer:** {entry['answer']}")
            st.markdown("---")

    # Button to reset conversation
    if st.button("Reset Conversation"):
        st.session_state.conversation_history = []
        st.session_state.database_initialized = False
        clear_database()
        st.experimental_rerun()
else:
    st.info("👆 Start by uploading a PDF file.")

# Sidebar instructions
with st.sidebar:
    st.markdown("""
    ### How to use this app:
    1. Upload a PDF file
    2. Wait for the system to process it
    3. Ask questions about the content
    4. View answers and sources
    5. Use the Reset button to start over
    """)
