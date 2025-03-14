# DB-LLM: Natural Language Database Query System

A system that allows users to query databases using natural language. This project integrates:

1. Database connections via SQLAlchemy
2. Vector database storage (Qdrant) for metadata
3. OpenAI API for embeddings and natural language understanding

## Architecture

- `src/db/`: Database connection and query functionality
- `src/vector_store/`: Vector database integration for storing DB metadata
- `src/llm/`: LLM integration for natural language understanding
- `config/`: Configuration files
- `data/`: Data storage directory

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the root directory with the following variables:
   ```
   DB_HOST=62.90.141.45
   DB_PORT=2222
   DB_USER=behatsdaa
   DB_PASSWORD=Bb4H68LkHG98qEhKjH2A
   DB_NAME=behatsdaa
   
   OPENAI_API_KEY=your_openai_api_key
   
   VECTOR_DB_PATH=./data/vector_db
   ```
   
   **Note**: This project uses OpenAI's API for both embeddings and completions, so a valid API key is required.

3. Initialize the database schema:
   ```bash
   python src/scripts/index_database.py
   ```

4. Run the query interface:
   ```bash
   python src/main.py
   ```

## Usage

Users can interact with the system by:
1. Entering natural language queries
2. The system converts these to SQL
3. Executes the query against the database
4. Returns formatted results

## Example Queries

- "Show me sales statistics from 2015"
- "What were the top 5 customers by revenue last month?"
- "Generate a report of inventory items below reorder threshold"

For detailed installation instructions, see [INSTALL.md](INSTALL.md) 