#!/usr/bin/env python3
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Suppress excessive logging
logging.basicConfig(level=logging.ERROR)  # Only show errors
# Silence specific loggers that are too verbose
for logger_name in ['httpx', 'src.db.sql_file_parser', 'src.vector_store.embeddings']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Add the project directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.db.sql_file_parser import SQLFileParser
from src.vector_store.embeddings import SchemaEmbedder
from src.vector_store.vector_db import VectorStore

def init_vector_db():
    """Initialize the vector database with all available tables"""
    load_dotenv()
    
    # Find most recent SQL file in data directory
    data_dir = Path("data")
    sql_files = list(data_dir.glob("*.sql.gz")) + list(data_dir.glob("*.sql"))
    if not sql_files:
        print("No SQL files found in data directory")
        return False
    
    sql_file = sorted(sql_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    print(f"Using SQL file: {sql_file.name}")
    
    # Initialize components
    sql_parser = SQLFileParser(str(sql_file))
    embedder = SchemaEmbedder()
    vector_store = VectorStore()
    
    # Get all tables
    all_tables = sql_parser.get_all_table_names()
    if not all_tables:
        print("No tables found in the SQL file")
        return False
    
    print(f"Processing {len(all_tables)} tables...")
    
    # Clear existing collection
    vector_store.clear_collection()
    
    # Generate and store embeddings
    all_embeddings = []
    processed_count = 0
    
    for table_name in all_tables:
        # Simple progress indicator
        processed_count += 1
        if processed_count % 10 == 0 or processed_count == len(all_tables):
            print(f"Progress: {processed_count}/{len(all_tables)} tables", end="\r")
        
        vector_chunks = sql_parser.create_vector_chunks(table_name, limit=1)
        
        if vector_chunks:
            chunk = vector_chunks[0]
            embedding = embedder.embed_text(chunk["content"])
            
            all_embeddings.append({
                "vector": embedding,
                "payload": {
                    "type": "table_with_samples",
                    "name": table_name,
                    "table": table_name,
                    "content": chunk["content"],
                    "metadata": chunk["metadata"]
                }
            })
    
    print()  # New line after progress indicator
    
    # Store the embeddings
    if all_embeddings:
        vector_store.store_embeddings(all_embeddings)
        print(f"Stored {len(all_embeddings)} table embeddings in vector database")
        return True
    
    return False

if __name__ == "__main__":
    print("Initializing vector database...")
    success = init_vector_db()
    
    if success:
        print("✓ Vector database initialization complete!")
    else:
        print("✗ Vector database initialization failed!") 