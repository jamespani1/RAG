
import os
import shutil
import logging
from fastapi import (
    FastAPI, 
    UploadFile, 
    File, 
    HTTPException, 
    Request,
    status
)
from fastapi.responses import JSONResponse

from models.models import QueryRequest,QueryResponse,UploadResponse,ErrorResponse
from core.config import UPLOAD_DIRECTORY, VECTOR_STORE_DIRECTORY
from services import rag_service


app = FastAPI(
    title="PDF RAG API",
    description="An API to upload PDFs and query them using a local RAG model.",
    version="1.0.0"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.on_event("startup")
def on_startup():
    """Create necessary directories on application startup."""
    try:
        os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
        os.makedirs(VECTOR_STORE_DIRECTORY, exist_ok=True)
        logger.info(f"Created directories: {UPLOAD_DIRECTORY}, {VECTOR_STORE_DIRECTORY}")
    except Exception as e:
        logger.critical(f"Failed to create directories on startup: {e}", exc_info=True)
        raise RuntimeError(f"Startup directory creation failed: {e}")



@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom handler for FastAPI's HTTPExceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Custom handler for all unhandled exceptions."""
    logger.error(f"Unhandled exception for {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"An internal server error occurred: {str(exc)}"}
    )


@app.post(
    "/upload-pdf/", 
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF for processing",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Uploads a PDF file, saves it, and processes it into the vector store.
    """
    if not file.filename.endswith(".pdf"):
        logger.warning(f"Failed upload attempt: Non-PDF file '{file.filename}'")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF files are allowed."
        )

    secure_filename = os.path.basename(file.filename)
    file_path = os.path.join(UPLOAD_DIRECTORY, secure_filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Successfully saved file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save file {file_path}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {e}"
        )
    finally:
        file.file.close()

    try:
        success = rag_service.load_pdf_to_vector_store(file_path, secure_filename)
        if not success:
            logger.warning(f"PDF {secure_filename} was processed but yielded no documents.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The PDF file appears to be empty or corrupted."
            )
            
        return UploadResponse(
            message="File processed and added to vector store successfully.",
            file_name=secure_filename
        )
    
    except ValueError as ve:
        logger.error(f"Failed to process PDF {secure_filename}: {ve}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process PDF: {ve}"
        )
    except Exception as e:
        logger.error(f"Failed to load PDF {secure_filename} to vector store: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during file processing: {e}"
        )

@app.post(
    "/query/", 
    response_model=QueryResponse,
    summary="Query a previously uploaded PDF",
    responses={
        404: {"model": ErrorResponse, "description": "File not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def query_pdf(request: QueryRequest):
    """
    Receives a query and a file_name, then returns a RAG-generated answer
    based on the content of that specific file.
    """
    
    file_path = os.path.join(UPLOAD_DIRECTORY, request.file_name)
    if not os.path.exists(file_path):
        logger.warning(f"Query attempt on non-existent file: {request.file_name}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {request.file_name}. Please upload it first."
        )

    try:
        answer = rag_service.query_rag(request.query, request.file_name)
        
        return QueryResponse(
            answer=answer,
            file_name=request.file_name
        )
    except Exception as e:
        logger.error(f"Error during RAG query for file {request.file_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while generating the answer: {e}"
        )