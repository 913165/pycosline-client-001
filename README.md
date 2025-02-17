# pycosline-client-001
# RAG Service with Mistral AI

A FastAPI service that implements Retrieval-Augmented Generation (RAG) using Mistral AI for embeddings and vector storage integration.

## Features

- 📝 Document Processing
  - Text file upload and processing
  - CSV file support
  - Batch processing capability

- 🧠 Embeddings Generation
  - Mistral AI embeddings
  - Single text embedding
  - Batch embeddings
  - UUID tracking for each embedding

- 🔄 Vector Storage Integration
  - Vector database API integration
  - Batch payload upload
  - Point-based storage format

- 🔍 Query Processing
  - RAG-based querying
  - Context-aware responses
  - Source tracking

## Prerequisites

- Python 3.9+
- Mistral AI API Key
- Vector Database (running on localhost:8000)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-service
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create .env file:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

## Project Structure

```
app/
├── __init__.py
├── main.py
├── config/
│   ├── __init__.py
│   └── settings.py
├── models/
│   ├── __init__.py
│   ├── request_models.py
│   └── response_models.py
├── services/
│   ├── __init__.py
│   └── rag_processor.py
├── routers/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── document.py
│   └── health.py
└── utils/
    ├── __init__.py
    └── exceptions.py
```

## API Endpoints

### Document Upload and Processing

```bash
# Upload and process a file
POST /process-file
Content-Type: multipart/form-data
file: your_file.txt
```

Response:
```json
{
    "status": "success",
    "processed_points": 10,
    "vector_db_response": {...}
}
```

### Embeddings Generation

```bash
# Get embedding for single text
POST /embeddings
Content-Type: application/json
{
    "text": "Your text here"
}
```

Response:
```json
{
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "text": "Your text here",
    "embedding": [...],
    "metadata": {
        "source": "single_request",
        "id": "123e4567-e89b-12d3-a456-426614174000"
    }
}
```

```bash
# Batch embeddings
POST /batch-embeddings
Content-Type: application/json
{
    "texts": ["Text 1", "Text 2"],
    "batch_size": 32
}
```

### Query Processing

```bash
POST /query
Content-Type: application/json
{
    "question": "Your question here"
}
```

Response:
```json
{
    "answer": "Generated answer",
    "sources": ["source1", "source2"]
}
```

## Running the Service

1. Start the service:
```bash
uvicorn app.main:app --reload --port 8000
```

2. Access the API documentation:
```
http://localhost:8000/docs
```

## Vector Database Integration

The service integrates with a vector database running on:
```
http://localhost:8000/api/v1/collections/vector_store
```

Points are stored in the following format:
```json
{
    "id": "uuid",
    "payload": {
        "content": "original text",
        "embedding": [float values],
        "metadata": {
            "source": "file_upload",
            "id": "uuid"
        }
    }
}
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| MISTRAL_API_KEY | Mistral AI API Key | Yes |

## Testing

Run tests using pytest:
```bash
pytest tests/
```

## Error Handling

The service includes comprehensive error handling for:
- File upload errors
- Embedding generation failures
- Vector database communication issues
- Invalid input formats

## Limitations

- Currently supports only .txt and .csv files
- Maximum file size: 10MB
- Batch size limit: 32 items
- Vector database must be running locally

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[MIT License](LICENSE)