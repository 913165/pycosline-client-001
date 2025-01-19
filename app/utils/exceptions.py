# app/utils/exceptions.py
from fastapi import HTTPException

class FileTypeError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=400, detail=detail)

class ProcessingError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=500, detail=detail)

class VectorDBError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=500, detail=detail)