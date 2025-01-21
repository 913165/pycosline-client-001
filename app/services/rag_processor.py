# app/services/rag_processor.py
from uuid import uuid4
from fastapi import HTTPException
from typing import List, Dict, Optional, Tuple
from io import StringIO

from langchain_community.document_loaders import TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

from app.config.settings import MISTRAL_API_KEY
from app.models.request_models import ChatMessage
from app.models.response_models import Point
import logging
import traceback




class RAGProcessor:
    logging.basicConfig(level=logging.INFO)
    logging = logging.getLogger(__name__)
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
            loader = TextLoader(file_path)
            docs = loader.load()
            documents = self.text_splitter.split_documents(docs)
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

    async def get_batch_embeddings(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        try:
            embeddings = []
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
            raise HTTPException(
                status_code=500,
                detail=f"Error querying document: {str(e)}"
            )

    async def process_file_for_embeddings(self, file_content: bytes) -> List[Point]:
        try:
            lines = [line.strip() for line in StringIO(file_content.decode('utf-8')).readlines() if line.strip()]
            embeddings = await self.get_batch_embeddings(lines)
            points = []
            for text, embedding_data in zip(lines, embeddings):
                point_id = uuid4()
                point = Point(
                    id=point_id,
                    content=text,
                    embedding=embedding_data["embedding"],
                    metadata={
                        "source": "file_upload",
                        "id": str(point_id)
                    },
                    media=[]
                )
                points.append(point)
            return points
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing file: {str(e)}"
            )

    chat_prompt = ChatPromptTemplate.from_template("""
           Answer the following question based on the provided context and chat history.

           Previous Conversation:
           {chat_history}

           Context from documents:
           {context}

           Current Question: {input}

           Please provide a clear and contextual response.
    """)

    # Add this to your RAGProcessor class

    # Add this to your RAGProcessor class

    async def process_chat(
            self,
            question: str,
            chat_history: Optional[List[ChatMessage]] = None,
            similarity_threshold: float = 0.7,
            top_k: int = 5,
            collection_name: str = "vector_store"
    ) -> Tuple[str, List[str], List[float]]:
        """
        Process chat using existing similarity search functionality.
        """
        from app.routers.similarity import SearchRequest, similarity_search
        try:
            # Create search request using existing SearchRequest model
            search_request = SearchRequest(
                query=question,
                topK=top_k,
                similarityThreshold=similarity_threshold
            )

            # Use existing similarity search to get relevant documents
            search_response = await similarity_search(
                request=search_request,
                collection_name=collection_name
            )

            # Format chat history
            formatted_history = ""
            if chat_history:
                formatted_history = "\n".join([
                    f"{msg.role}: {msg.content}"
                    for msg in chat_history[-5:]
                ])

            # Create the message content
            message_content = f"""
            Previous conversation:
            {formatted_history}

            Context from documents:
            {' '.join(search_response.content)}

            Question: {question}

            Please provide a helpful response based on the context provided.
            """

            # Use LangChain's ChatMistralAI interface
            response = await self.model.ainvoke(message_content)

            # Return answer with documents as sources
            return response.content, search_response.content, [1.0] * len(search_response.content)

        except Exception as e:
            logging.error(f"Error processing chat: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error in chat processing: {str(e)}"
            )
# Initialize a single instance
rag_processor = RAGProcessor()