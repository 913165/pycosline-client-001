# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import ALLOWED_ORIGINS
from app.routers import document, embeddings, health, similarity

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(document.router, tags=["Documents"])
app.include_router(embeddings.router, tags=["Embeddings"])
app.include_router(health.router, tags=["Health"])

#  router for similarity
app.include_router(similarity.router, tags=["Similarity"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000,reload=True)