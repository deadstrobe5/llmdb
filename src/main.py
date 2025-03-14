#!/usr/bin/env python3
import os
import sys
import logging
import argparse
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

from db.connector import DatabaseConnector
from vector_store.embeddings import SchemaEmbedder
from vector_store.vector_db import VectorStore
from llm.nl_to_sql import NLToSQL

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NLDatabaseInterface:
    """
    Main interface for natural language to database queries.
    """
    
    def __init__(self, init_db: bool = False, offline_mode: bool = False, force_init: bool = False):
        """
        Initialize the NL database interface.
        
        Args:
            init_db: Whether to initialize/update the vector database with schema information
            offline_mode: Whether to run in offline mode (print SQL instead of executing)
            force_init: Whether to force reinitialization of vector database even if embeddings exist
        """
        load_dotenv()
        
        # Check if OpenAI API key is set
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY is not set in environment variables")
            print("Error: OPENAI_API_KEY is not set in environment variables")
            print("Please add your OpenAI API key to the .env file")
            sys.exit(1)
        
        # Initialize components
        self.offline_mode = offline_mode
        self.force_init = force_init
        
        if not offline_mode:
            self.db_connector = DatabaseConnector()
        else:
            logger.info("Running in offline mode - SQL queries will be printed instead of executed")
            self.db_connector = None
            
        self.embedder = SchemaEmbedder()
        self.vector_store = VectorStore()
        self.nl_to_sql = NLToSQL()
        
        # Initialize vector DB if requested
        if init_db:
            self.initialize_vector_db()
    
    def initialize_vector_db(self) -> None:
        """Initialize the vector database with schema information"""
        try:
            # Check if embeddings already exist and we're not forcing reinitialization
            if not self.force_init and self.vector_store.has_embeddings():
                logger.info("Vector database already has embeddings. Skipping initialization.")
                print("Vector database already has embeddings. Use --force-init to reinitialize.")
                return
                
            logger.info("Extracting database schema...")
            
            if not self.offline_mode:
                schema = self.db_connector.get_full_database_schema()
            else:
                # In offline mode, use the SQL file parser
                from db.sql_file_parser import SQLFileParser
                
                # Find the most recent .sql.gz file in the data directory
                data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
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
            
            logger.info("Generating embeddings for schema...")
            if self.offline_mode:
                # Pass the SQL parser to extract sample data
                embeddings = self.embedder.generate_schema_embeddings(
                    schema,
                    sql_file_parser=sql_parser,
                    sample_data_limit=3
                )
            else:
                # In online mode, we don't have a SQL parser
                embeddings = self.embedder.generate_schema_embeddings(schema)
            
            logger.info("Storing embeddings in vector database...")
            self.vector_store.clear_collection()
            self.vector_store.store_table_embeddings(embeddings)
            
            logger.info("Vector database initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {str(e)}")
            raise
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query.
        
        Args:
            query: Natural language query string
            
        Returns:
            Dictionary containing results and metadata
        """
        try:
            # 1. Generate embedding for query
            logger.info(f"Processing query: {query}")
            query_embedding = self.embedder.embed_query(query)
            
            # 2. Search vector database for relevant schema elements
            logger.info("Searching for relevant schema elements...")
            relevant_schemas = self.vector_store.search_schema(query_embedding, limit=5)
            
            # Log detailed information about what the vector database returned
            logger.info(f"Vector database returned {len(relevant_schemas)} relevant schema elements:")
            for i, schema in enumerate(relevant_schemas):
                logger.info(f"  Result {i+1}:")
                logger.info(f"    ID: {schema['id']}")
                logger.info(f"    Score: {schema['score']}")
                logger.info(f"    Type: {schema['payload']['type']}")
                if schema['payload']['type'] == 'table':
                    logger.info(f"    Table: {schema['payload']['name']}")
                else:
                    logger.info(f"    Table: {schema['payload']['table']}")
                    logger.info(f"    Column: {schema['payload']['name']}")
            
            # 3. Generate SQL from natural language
            logger.info("Converting natural language to SQL...")
            sql_result = self.nl_to_sql.nl_to_sql(query, relevant_schemas)
            
            # 4. Execute SQL query if valid
            if sql_result["sql"]:
                if not self.offline_mode:
                    # Online mode: Execute the query
                    logger.info(f"Executing SQL query: {sql_result['sql']}")
                    try:
                        results, columns = self.db_connector.execute_query(sql_result["sql"])
                        return {
                            "query": query,
                            "sql": sql_result["sql"],
                            "explanation": sql_result["explanation"],
                            "results": results,
                            "columns": columns,
                            "success": True,
                            "error": None
                        }
                    except Exception as e:
                        logger.error(f"SQL execution failed: {str(e)}")
                        return {
                            "query": query,
                            "sql": sql_result["sql"],
                            "explanation": sql_result["explanation"],
                            "results": [],
                            "columns": [],
                            "success": False,
                            "error": str(e)
                        }
                else:
                    # Offline mode: Just return the SQL without executing
                    logger.info(f"Offline mode - SQL query generated: {sql_result['sql']}")
                    return {
                        "query": query,
                        "sql": sql_result["sql"],
                        "explanation": sql_result["explanation"],
                        "results": [],
                        "columns": [],
                        "success": True,
                        "error": None,
                        "offline_mode": True
                    }
            else:
                logger.warning("Failed to generate valid SQL")
                return {
                    "query": query,
                    "sql": "",
                    "explanation": sql_result["explanation"],
                    "results": [],
                    "columns": [],
                    "success": False,
                    "error": "Failed to generate valid SQL"
                }
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return {
                "query": query,
                "sql": "",
                "explanation": "",
                "results": [],
                "columns": [],
                "success": False,
                "error": str(e)
            }

def format_results(results: List[Dict[str, Any]], columns: List[str]) -> str:
    """Format query results for display"""
    if not results:
        return "No results found."
    
    # Format as table
    output = []
    
    # Header
    header = " | ".join(columns)
    separator = "-" * len(header)
    output.append(header)
    output.append(separator)
    
    # Rows
    for row in results:
        row_values = [str(row.get(col, "")) for col in columns]
        output.append(" | ".join(row_values))
    
    return "\n".join(output)

def interactive_mode(interface: NLDatabaseInterface) -> None:
    """Run in interactive mode"""
    print("\n=== Natural Language Database Query Interface ===")
    print("Enter 'exit' or 'quit' to exit, 'init' to initialize/update vector DB\n")
    
    while True:
        query = input("\nEnter your query: ")
        
        if query.lower() in ["exit", "quit"]:
            break
        elif query.lower() == "init":
            try:
                interface.initialize_vector_db()
                print("Vector database initialized successfully")
            except Exception as e:
                print(f"Failed to initialize vector database: {str(e)}")
            continue
        
        # Process query
        result = interface.process_query(query)
        
        # Display results
        if result["success"]:
            print("\n=== SQL Query ===")
            print(result["sql"])
            print("\n=== Explanation ===")
            print(result["explanation"])
            
            if result.get("offline_mode"):
                print("\n=== OFFLINE MODE ===")
                print("Query not executed. SQL query is shown above.")
            else:
                print("\n=== Results ===")
                print(format_results(result["results"], result["columns"]))
        else:
            print("\n=== Error ===")
            print(result["error"])
            if result["explanation"]:
                print("\n=== Explanation ===")
                print(result["explanation"])

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Natural Language Database Query Interface")
    parser.add_argument("--init", action="store_true", help="Initialize/update vector database")
    parser.add_argument("--query", type=str, help="Execute a single query and exit")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode (print SQL instead of executing)")
    parser.add_argument("--force-init", action="store_true", help="Force reinitialization of vector database even if embeddings exist")
    args = parser.parse_args()
    
    try:
        interface = NLDatabaseInterface(init_db=args.init, offline_mode=args.offline, force_init=args.force_init)
        
        if args.query:
            # Single query mode
            result = interface.process_query(args.query)
            
            if result["success"]:
                print(result["sql"])
                if not args.offline:
                    print(format_results(result["results"], result["columns"]))
                else:
                    print("\nOFFLINE MODE: Query not executed")
            else:
                print(f"Error: {result['error']}")
                sys.exit(1)
        else:
            # Interactive mode
            interactive_mode(interface)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 