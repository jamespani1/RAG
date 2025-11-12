from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    file_name: str

class UploadResponse(BaseModel):
    message: str
    file_name: str

class QueryResponse(BaseModel):
    answer: str
    file_name: str

class ErrorResponse(BaseModel):
    detail: str