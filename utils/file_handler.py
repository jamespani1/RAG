
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_community.docstore.document import Document
import logging

from core.config import CHUNK_SIZE, CHUNK_OVERLAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_pdf(file_path: str) -> List[Document]:
    """
    Loads a PDF, splits it into chunks, and returns a list of Documents.
    
    Args:
        file_path (str): The path to the PDF file.

    Returns:
        List[Document]: A list of document chunks.
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        if not documents:
            logger.warning(f"No documents loaded from {file_path}. The PDF might be empty or corrupted.")
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        split_docs = text_splitter.split_documents(documents)
        
        logger.info(f"Successfully processed {file_path}. Created {len(split_docs)} chunks.")
        
        return split_docs

    except Exception as e:
        logger.error(f"Error processing PDF file {file_path}: {e}", exc_info=True)
        raise ValueError(f"Failed to process PDF: {e}")