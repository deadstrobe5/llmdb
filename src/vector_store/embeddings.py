import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SchemaEmbedder:
    """
    Generates and manages embeddings for database schema elements using OpenAI API.
    """
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        Initialize the schema embedder.
        
        Args:
            model_name: Name of the OpenAI embedding model to use
        """
        load_dotenv()
        
        # Check if API key is set
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY is not set in environment variables")
            
        logger.info(f"Initializing schema embedder with OpenAI model: {model_name}")
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key)
        
    def _create_table_description(self, table_schema: Dict[str, Any], sample_data: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Create a textual description of a table schema with sample data.
        
        Args:
            table_schema: Dictionary containing table schema information
            sample_data: Optional list of dictionaries containing sample data
            
        Returns:
            String description of the table schema with sample data
        """
        table_name = table_schema["name"]
        description = f"TABLE: {table_name}\n"
        
        # Add column information
        description += "\nCOLUMNS:\n"
        primary_keys = []
        foreign_keys = {}

        # First pass to collect key information
        for column in table_schema.get("columns", []):
            if column.get("primary_key", False):
                primary_keys.append(column["name"])
                
        # Extract foreign key information
        if table_schema.get("foreign_keys"):
            for fk in table_schema["foreign_keys"]:
                const_cols = fk["constrained_columns"]
                ref_table = fk["referred_table"]
                ref_cols = fk["referred_columns"]
                for i, col in enumerate(const_cols):
                    foreign_keys[col] = {
                        "ref_table": ref_table,
                        "ref_column": ref_cols[i] if i < len(ref_cols) else ref_cols[0]
                    }
        
        # Format column descriptions with type and constraint information
        for column in table_schema.get("columns", []):
            col_name = column["name"]
            col_type = column["type"]
            nullable = "NULL" if column.get("nullable", True) else "NOT NULL"
            pk = "PRIMARY KEY" if column.get("primary_key", False) else ""
            default = f"DEFAULT {column['default']}" if column.get("default") else ""
            
            # Add relationship context
            relationship = ""
            if col_name in foreign_keys:
                ref = foreign_keys[col_name]
                relationship = f" - REFERENCES {ref['ref_table']}({ref['ref_column']})"
            
            col_desc = f"  - {col_name}: {col_type} {nullable} {pk} {default}{relationship}"            
            description += col_desc + "\n"
        
        # Add key information
        if primary_keys:
            description += f"\nPRIMARY KEY: {', '.join(primary_keys)}\n"
            
        # Add foreign key information in standard SQL format
        if foreign_keys:
            description += "\nFOREIGN KEYS:\n"
            for col, ref in foreign_keys.items():
                description += f"  - {col} -> {ref['ref_table']}({ref['ref_column']})\n"
        
        # Add indices information
        if table_schema.get("indices"):
            description += "\nINDICES:\n"
            for idx in table_schema.get("indices", []):
                idx_name = idx.get("name", "unnamed_index")
                idx_cols = ", ".join(idx.get("columns", []))
                unique = "UNIQUE " if idx.get("unique", False) else ""
                description += f"  - {unique}INDEX {idx_name} ({idx_cols})\n"
            
        # Add sample data if available
        if sample_data and len(sample_data) > 0:
            description += "\nSAMPLE DATA:\n"
            
            # Get column names from the first row
            if sample_data[0]:
                columns = list(sample_data[0].keys())
                
                # Create a table-like format for sample data
                # Header
                header = " | ".join(columns)
                separator = "-" * len(header)
                description += f"{header}\n{separator}\n"
                
                # Rows
                for row in sample_data:
                    row_values = []
                    for col in columns:
                        # Truncate long values
                        value = str(row.get(col, ""))
                        if len(value) > 20:
                            value = value[:17] + "..."
                        row_values.append(value)
                    
                    description += " | ".join(row_values) + "\n"
        
        return description
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a text using OpenAI API.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            # Return a zero vector as fallback (not ideal but prevents crashes)
            return [0.0] * 1536  # Default size for OpenAI embeddings
    
    def generate_table_embedding(self, table_schema: Dict[str, Any], sample_data: Optional[List[Dict[str, Any]]] = None) -> List[float]:
        """
        Generate an embedding for a table schema with sample data.
        
        Args:
            table_schema: Dictionary containing table schema information
            sample_data: Optional list of dictionaries containing sample data
            
        Returns:
            List of floats containing the embedding vector
        """
        description = self._create_table_description(table_schema, sample_data)
        return self._get_embedding(description)
    
    def generate_column_embedding(self, table_name: str, column: Dict[str, Any]) -> List[float]:
        """
        Generate an embedding for a table column.
        
        Args:
            table_name: Name of the table the column belongs to
            column: Dictionary containing column information
            
        Returns:
            List of floats containing the embedding vector
        """
        description = self._create_column_description(table_name, column)
        return self._get_embedding(description)
    
    def _create_column_description(self, table_name: str, column: Dict[str, Any]) -> str:
        """
        Create a textual description of a table column.
        
        Args:
            table_name: Name of the table the column belongs to
            column: Dictionary containing column information
            
        Returns:
            String description of the column
        """
        col_name = column["name"]
        col_type = column["type"]
        nullable = "NULL" if column.get("nullable", True) else "NOT NULL"
        pk = "PRIMARY KEY" if column.get("primary_key", False) else ""
        default = f"DEFAULT {column['default']}" if column.get("default") else ""
        
        return f"Table: {table_name}, Column: {col_name}, Type: {col_type}, {nullable} {pk} {default}"
    
    def print_table_descriptions(self, schema: Dict[str, Dict[str, Any]], sql_file_parser: Optional[Any] = None, sample_data_limit: int = 3) -> None:
        """
        Print table descriptions that would be used for embeddings.
        This is a debug function that doesn't actually generate embeddings.
        
        Args:
            schema: Dictionary containing schema information
            sql_file_parser: SQL file parser to extract sample data
            sample_data_limit: Maximum number of sample data rows to include
        """
        for table_name, table_schema in schema.items():
            print(f"\n{'='*80}\n")
            
            # Extract sample data if parser is provided
            sample_data = None
            if sql_file_parser:
                try:
                    sample_data = sql_file_parser.extract_sample_data(table_name, limit=sample_data_limit)
                except Exception as e:
                    print(f"Failed to extract sample data: {str(e)}")
            
            description = self._create_table_description(table_schema, sample_data)
            print(description)
            print(f"\n{'='*80}\n")
    
    def generate_schema_embeddings(self, 
                                  full_schema: Dict[str, Dict[str, Any]],
                                  sql_file_parser: Optional[Any] = None,
                                  sample_data_limit: int = 3,
                                  debug_mode: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Generate embeddings for all tables in a database schema.
        
        Args:
            full_schema: Dictionary containing full database schema
            sql_file_parser: Optional SQLFileParser instance to extract sample data
            sample_data_limit: Maximum number of sample data rows to include
            debug_mode: If True, print descriptions instead of generating embeddings
            
        Returns:
            Dictionary containing tables and their embeddings
        """
        embeddings = {}
        
        if debug_mode:
            # Just print the descriptions without generating embeddings
            self.print_table_descriptions(full_schema, sql_file_parser, sample_data_limit)
            return {}
        
        for table_name, table_schema in full_schema.items():
            logger.info(f"Generating embeddings for table: {table_name}")
            
            # Extract sample data if parser is provided
            sample_data = None
            if sql_file_parser:
                try:
                    sample_data = sql_file_parser.extract_sample_data(table_name, limit=sample_data_limit)
                    if sample_data:
                        logger.info(f"Extracted {len(sample_data)} sample data rows for table: {table_name}")
                    else:
                        logger.info(f"No sample data found for table: {table_name}")
                except Exception as e:
                    logger.warning(f"Failed to extract sample data for table {table_name}: {str(e)}")
            
            # Generate table embedding
            description = self._create_table_description(table_schema, sample_data)
            table_embedding = self._get_embedding(description)
            
            # Store embedding and description for reference
            embeddings[table_name] = {
                "table_embedding": table_embedding,
                "description": description,
                "schema": table_schema
            }
        
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate an embedding for a user query.
        
        Args:
            query: User's natural language query
            
        Returns:
            List of floats containing the embedding vector
        """
        return self._get_embedding(query)
        
    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding for any text content.
        Useful for embedding sample data or other textual content.
        
        Args:
            text: Text content to embed
            
        Returns:
            List of floats containing the embedding vector
        """
        return self._get_embedding(text) 