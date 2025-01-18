# main.py
from uuid import UUID, uuid4
from fastapi import FastAPI, UploadFile, HTTPException, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Union, Any
import os
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class QueryRequest(BaseModel):
    question: str
    context_filter: Optional[dict] = None


class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = None


class ProcessingConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    model_name: str = "mistral-embed"


class EmbeddingRequest(BaseModel):
    text: str


class EmbeddingResponse(BaseModel):
    id: UUID  # Add UUID field
    text: str
    embedding: List[float]
    metadata: Optional[Dict[str, Any]] = None


class BatchEmbeddingRequest(BaseModel):
    texts: List[str]
    batch_size: Optional[int] = 32


class BatchEmbeddingResponse(BaseModel):
    embeddings: List[Dict[str, Union[str, List[float]]]]


class Point(BaseModel):
    id: UUID
    content: str
    embedding: List[float]
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class RAGProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=MISTRAL_API_KEY
        )
        self.model = ChatMistralAI(mistral_api_key=MISTRAL_API_KEY)
        self.vector_store = None
        self.prompt = ChatPromptTemplate.from_template("""
            Answer the following question based only on the provided context:
            <context>
            {context}
            </context>
            Question: {input}
        """)

    async def process_document(self, file_path: str):
        try:
            # Load and process document
            loader = TextLoader(file_path)
            docs = loader.load()
            documents = self.text_splitter.split_documents(docs)

            # Create vector store
            self.vector_store = FAISS.from_documents(documents, self.embeddings)

            return {
                "status": "success",
                "message": "Document processed successfully",
                "num_chunks": len(documents)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_embeddings(self, text: str) -> List[float]:
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating embeddings: {str(e)}"
            )

    async def query(self, question: str) -> str:
        if not self.vector_store:
            raise HTTPException(
                status_code=400,
                detail="No documents have been processed yet"
            )

        try:
            retriever = self.vector_store.as_retriever()
            document_chain = create_stuff_documents_chain(self.model, self.prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            response = retrieval_chain.invoke({"input": question})
            return response["answer"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_batch_embeddings(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        try:
            embeddings = []
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.embeddings.embed_documents(batch)
                for text, embedding in zip(batch, batch_embeddings):
                    embeddings.append({
                        "text": text,
                        "embedding": embedding
                    })
            return embeddings
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating batch embeddings: {str(e)}"
            )


# Initialize RAG processor
rag_processor = RAGProcessor()


@app.post("/upload")
async def upload_document(
        file: UploadFile = File(...),
        config: Optional[ProcessingConfig] = None
):
    if not file.filename.endswith('.txt'):
        raise HTTPException(
            status_code=400,
            detail="Only .txt files are currently supported"
        )

    try:
        # Save uploaded file temporarily
        file_path = f"temp/{file.filename}"
        os.makedirs("temp", exist_ok=True)

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Process document
        result = await rag_processor.process_document(file_path)

        # Cleanup
        os.remove(file_path)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    try:
        answer = await rag_processor.query(request.question)
        return QueryResponse(
            answer=answer,
            sources=None  # You could add source tracking if needed
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    try:
        embedding = await rag_processor.get_embeddings(request.text)
        point_id = uuid4()  # Generate UUID
        return EmbeddingResponse(
            text=request.text,
            embedding=embedding,
            id=point_id,  # Include UUID in response
            metadata={
                "source": "single_request",
                "id": str(point_id)  # Include UUID in metadata as well
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-embeddings", response_model=BatchEmbeddingResponse)
async def get_batch_embeddings(request: BatchEmbeddingRequest):
    try:
        embeddings = await rag_processor.get_batch_embeddings(
            request.texts,
            request.batch_size or 32
        )
        return BatchEmbeddingResponse(embeddings=embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vector_store_initialized": rag_processor.vector_store is not None,
        "embedding_model": "mistral-embed",
        "api_key_configured": bool(MISTRAL_API_KEY)
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)