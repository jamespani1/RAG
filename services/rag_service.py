
import logging
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from core.config import (
    VECTOR_STORE_DIRECTORY, 
    EMBEDDING_MODEL_NAME,
    GROQ_API_KEY,  
    GROQ_MODEL_NAME
)
from utils.file_handler import process_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


try:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'} 
    )

    vector_store = Chroma(
        persist_directory=VECTOR_STORE_DIRECTORY,
        embedding_function=embeddings
    )

    if not GROQ_API_KEY or GROQ_API_KEY == "groq_api_key":
        raise ValueError("GROQ_API_KEY is not set in core/config.py. Please get a free key from groq.com")

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL_NAME,
        temperature=0  
    )
    
    RAG_PROMPT_TEMPLATE = """
    Answer the user's question based only on the following context.
    If the context doesn't contain the answer, state "I don't have enough information from the document to answer that."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

except Exception as e:
    logger.critical(f"Failed to initialize RAG components: {e}", exc_info=True)
    raise RuntimeError(f"RAG Service Initialization Failed: {e}")


def format_docs(docs: list) -> str:
    """Combines document chunks into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def load_pdf_to_vector_store(file_path: str, file_name: str) -> bool:
    """
    Processes a PDF file and adds its content to the vector store.
    
    Args:
        file_path (str): The full path to the saved PDF file.
        file_name (str): The original name of the file (used for filtering).

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        docs = process_pdf(file_path)
        
        if not docs:
            logger.warning(f"No documents were created from {file_name}. Skipping vector store.")
            return False

        for doc in docs:
            doc.metadata["source"] = file_name
            
        vector_store.add_documents(docs)
        
        logger.info(f"Successfully added {len(docs)} chunks from {file_name} to vector store.")
        return True

    except Exception as e:
        logger.error(f"Error loading PDF {file_name} to vector store: {e}", exc_info=True)
        raise

def query_rag(query: str, file_name: str) -> str:
    """
    Queries the RAG system based on a user query and a specific file.
    
    Args:
        query (str): The user's question.
        file_name (str): The file to search within.

    Returns:
        str: The generated answer.
    """
    try:
        retriever = vector_store.as_retriever(
            search_kwargs={"filter": {"source": file_name}}
        )

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        
        logger.info(f"Invoking RAG chain for file '{file_name}' with query: '{query}'")
        
        answer = rag_chain.invoke(query)
        
        return answer

    except Exception as e:
        logger.error(f"Error during RAG query: {e}", exc_info=True)
        raise