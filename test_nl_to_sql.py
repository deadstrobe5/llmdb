#!/usr/bin/env python3
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add the project directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.vector_store.embeddings import SchemaEmbedder
from src.vector_store.vector_db import VectorStore
from src.llm.nl_to_sql import NLToSQL

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_nl_to_sql_conversion(query):
    """
    Test the full natural language to SQL conversion
    
    Args:
        query: Natural language query to convert to SQL
    """
    try:
        # Initialize components
        load_dotenv()
        embedder = SchemaEmbedder()
        vector_store = VectorStore()
        nl_to_sql = NLToSQL()
        
        # Check if vector database has embeddings
        if not vector_store.has_embeddings():
            logger.error("Vector database has no embeddings. Run ./init_db_sample.py first.")
            return None
        
        # Generate embedding for query
        logger.info(f"Generating embedding for query: {query}")
        query_embedding = embedder.embed_query(query)
        
        # Search vector database
        logger.info("Searching for relevant schema elements...")
        relevant_schemas = vector_store.search_schema(query_embedding, limit=5)
        
        logger.info(f"Found {len(relevant_schemas)} relevant schema elements:")
        for i, schema in enumerate(relevant_schemas):
            # Handle different payload types (schema vs sample data)
            if schema['payload']['type'] == 'sample_data':
                logger.info(f"  Result {i+1} (score: {schema['score']:.4f}): Sample data from {schema['payload']['table']}")
            else:
                logger.info(f"  Result {i+1} (score: {schema['score']:.4f}): {schema['payload'].get('name', 'Unknown')}")
        
        # Print full descriptions of the retrieved schemas
        print("\n=== RELEVANT SCHEMA INFORMATION ===")
        for i, schema in enumerate(relevant_schemas):
            payload = schema['payload']
            if 'content' in payload:
                print(f"\nSchema {i+1}: {payload.get('name', payload.get('table', 'Unknown'))} (score: {schema['score']:.4f})")
                print(f"Content:\n{payload['content']}")
            else:
                print(f"\nSchema {i+1}: {payload.get('name', payload.get('table', 'Unknown'))} (score: {schema['score']:.4f})")
                print(f"Description: No description")
        
        # Convert NL to SQL
        logger.info("Converting natural language to SQL...")
        sql_result = nl_to_sql.nl_to_sql(query, relevant_schemas)
        
        logger.info(f"Generated SQL: {sql_result['sql']}")
        logger.info(f"Explanation: {sql_result['explanation']}")
        
        return sql_result
    
    except Exception as e:
        logger.error(f"Error in NL to SQL conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Define test queries including some that would benefit from relationship information
    test_queries = [
        "Show me all users",
        "List all content items with their categories",
        "Show me the content items and their associated tags",
        "Find modules with their positions and titles",
        "What content belongs to each category?",
        "Show users who created content"
    ]
    
    # Allow command line arguments for custom queries
    if len(sys.argv) > 1:
        test_queries = [" ".join(sys.argv[1:])]
    
    print("\nTesting Natural Language to SQL conversion with relationship awareness:\n")
    for query in test_queries:
        print(f"\nQUERY: {query}")
        sql_result = test_nl_to_sql_conversion(query)
        
        if sql_result and sql_result["sql"]:
            print("\nGENERATED SQL:")
            print(f"{sql_result['sql']}")
            print("\nEXPLANATION:")
            print(f"{sql_result['explanation']}")
        else:
            print("Failed to generate SQL.")
        
        print("\n" + "="*80) 