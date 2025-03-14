#!/usr/bin/env python3
import os
import sys
import logging
import argparse
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Add src directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.connector import DatabaseConnector
from db.sql_file_parser import SQLFileParser
from vector_store.embeddings import SchemaEmbedder
from vector_store.vector_db import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def index_database(clear_existing: bool = True, use_sql_file: bool = False, sql_file_path: Optional[str] = None) -> None:
    """
    Extract database schema and store embeddings in vector database.
    
    Args:
        clear_existing: Whether to clear existing vector database collection
        use_sql_file: Whether to use a SQL file instead of connecting to the database
        sql_file_path: Path to the SQL file (if use_sql_file is True)
    """
    try:
        load_dotenv()
        
        logger.info("Initializing components...")
        
        # Get schema either from database or SQL file
        if use_sql_file:
            if not sql_file_path:
                # Find the most recent .sql.gz file in the data directory
                data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
                sql_files = [f for f in os.listdir(data_dir) if f.endswith('.sql.gz')]
                if not sql_files:
                    logger.error("No .sql.gz files found in the data directory")
                    raise FileNotFoundError("No .sql.gz files found in the data directory")
                
                # Sort by modification time (newest first)
                sql_files.sort(key=lambda f: os.path.getmtime(os.path.join(data_dir, f)), reverse=True)
                sql_file_path = os.path.join(data_dir, sql_files[0])
                
            logger.info(f"Using SQL file: {sql_file_path}")
            sql_parser = SQLFileParser(sql_file_path)
            schema = sql_parser.get_full_database_schema()
        else:
            logger.info("Connecting to database...")
            db_connector = DatabaseConnector()
            schema = db_connector.get_full_database_schema()
        
        logger.info(f"Found {len(schema)} tables")
        for table_name in schema.keys():
            logger.info(f"  - {table_name}")
        
        # Generate embeddings
        embedder = SchemaEmbedder()
        vector_store = VectorStore()
        
        logger.info("Generating embeddings for schema...")
        embeddings = embedder.generate_schema_embeddings(schema)
        
        if clear_existing:
            logger.info("Clearing existing vector database collection...")
            vector_store.clear_collection()
        
        logger.info("Storing embeddings in vector database...")
        vector_store.store_table_embeddings(embeddings)
        
        logger.info("Database indexing complete")
    except Exception as e:
        logger.error(f"Database indexing failed: {str(e)}")
        raise

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Index database schema for natural language querying")
    parser.add_argument("--keep-existing", action="store_true", help="Keep existing vector database entries")
    parser.add_argument("--use-sql-file", action="store_true", help="Use SQL file instead of connecting to database")
    parser.add_argument("--sql-file", type=str, help="Path to SQL file (if using --use-sql-file)")
    args = parser.parse_args()
    
    try:
        index_database(
            clear_existing=not args.keep_existing,
            use_sql_file=args.use_sql_file,
            sql_file_path=args.sql_file
        )
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 