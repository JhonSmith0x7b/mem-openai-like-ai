# mem-openai-like-ai

This project is a memory-augmented AI system built using OpenAI's API and a PostgreSQL-based vector database (`pgvector`). It enables the AI to retrieve and utilize user-specific memories to enhance responses.

## Features
- **Memory Integration**: Stores and retrieves user-specific memories using `pgvector`.
- **OpenAI Integration**: Leverages OpenAI's `text-embedding-3-large` model for embedding and chat completions.
- **Dockerized Setup**: Simplified deployment using Docker Compose.

## Project Structure
- `main.py`: Entry point for the application, defines the `YuKiNoAPI` class.
- `memory/beta_memory.py`: Implements memory management using `pgvector`.
- `.env`: Configuration file for environment variables.
- `docker-compose.yaml`: Docker Compose configuration for the application and PostgreSQL.

## Setup and Usage
1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd mem-openai-like-ai
    ```

2. Configure environment variables in `.env`:
    ```env
    PG_USER=postgres
    PG_PASSWORD=postgres
    PG_HOST=postgres
    PG_PORT=5432
    OPENAI_API_KEY=<your_openai_api_key>
    ```

3. Start the application using Docker Compose:
    ```bash
    docker-compose up --build
    ```

4. Access the application at `http://localhost:8086`.

## Key Components
- **Memory Management**: `Mem0Helper` class handles memory storage and retrieval.
- **API**: `YuKiNoAPI` class integrates memory into the AI's responses.
- **Database**: PostgreSQL with `pgvector` extension for vector storage.

## Example Workflow
1. User sends a message to the AI.
2. The system retrieves relevant memories using `Mem0Helper`.
3. Memories are injected into the AI's context for generating responses.
4. New interactions are stored as memories for future use.

## Requirements
- Docker and Docker Compose
- OpenAI API Key

## License
This project is licensed under the MIT License.  