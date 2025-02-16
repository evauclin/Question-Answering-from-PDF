import argparse
import os
import shutil
import logging

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import create_embedding_model
from langchain_chroma.vectorstores import Chroma

CHROMA_DIRECTORY = "chroma"
DATA_DIRECTORY = "data"

def load_pdf_documents() -> list[Document]:
    """
    Loads PDF documents from the DATA_DIRECTORY.

    :return: List of loaded PDF documents.
    """
    pdf_loader = PyPDFDirectoryLoader(DATA_DIRECTORY)
    return pdf_loader.load()

def split_pdf_documents(documents: list[Document]) -> list[Document]:
    """
    Splits PDF documents into chunks to facilitate processing.

    :param documents: List of PDF documents.
    :return: List of document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def assign_unique_chunk_ids(chunks: list[Document]) -> list[Document]:
    """
    Assigns a unique ID to each document chunk based on the source, page number,
    and chunk index.

    :param chunks: List of document chunks.
    :return: List of document chunks with unique IDs in their metadata.
    """
    previous_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown_source")
        page_number = chunk.metadata.get("page", "unknown_page")
        current_page_id = f"{source}:{page_number}"

        if current_page_id == previous_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        previous_page_id = current_page_id

    return chunks

def update_chroma_database(chunks: list[Document]):
    """
    Adds new document chunks to the Chroma database.
    Only chunks with IDs that do not already exist in the database are added.

    :param chunks: List of document chunks to add.
    """
    try:
        chroma_db = Chroma(
            persist_directory=CHROMA_DIRECTORY,
            embedding_function=create_embedding_model()
        )
    except Exception as error:
        logging.error("Error initializing the Chroma database: %s", error)
        return

    chunks_with_ids = assign_unique_chunk_ids(chunks)

    try:
        existing_items = chroma_db.get(include=[])
        existing_ids = set(existing_items["ids"])
    except Exception as error:
        logging.error("Error retrieving existing documents: %s", error)
        existing_ids = set()

    logging.info("Number of existing documents in the database: %d", len(existing_ids))

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata.get("id") not in existing_ids]

    if new_chunks:
        logging.info("Adding %d new document chunks", len(new_chunks))
        new_chunk_ids = [chunk.metadata.get("id", "Unknown ID") for chunk in new_chunks]
        chroma_db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        logging.info("No new document chunks to add")

def clear_database():
    """
    Deletes the Chroma persistence directory to reset the database.
    """
    if os.path.exists(CHROMA_DIRECTORY):
        shutil.rmtree(CHROMA_DIRECTORY)
        logging.info("Chroma database cleared.")
    else:
        logging.info("No existing Chroma database to clear.")

def main():
    parser = argparse.ArgumentParser(
        description="Update the vector database with PDF documents."
    )
    parser.add_argument("--reset", action="store_true", help="Reset the Chroma database.")
    args = parser.parse_args()

    if args.reset:
        logging.info("✨ Resetting the Chroma database.")
        clear_database()

    pdf_documents = load_pdf_documents()
    logging.info("Number of PDF documents loaded: %d", len(pdf_documents))
    document_chunks = split_pdf_documents(pdf_documents)
    logging.info("Number of document chunks generated: %d", len(document_chunks))
    update_chroma_database(document_chunks)

if __name__ == "__main__":
    main()
