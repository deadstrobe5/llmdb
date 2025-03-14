import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
import openai
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import OutputParserException
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SQLQueryResult(BaseModel):
    """Model for the SQL query result"""
    sql: str = Field(description="The SQL query to execute")
    explanation: str = Field(description="Explanation of what the SQL query does")

class NLToSQL:
    """
    Converts natural language queries to SQL using OpenAI.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the NL to SQL converter.
        
        Args:
            model_name: Name of the OpenAI model to use
        """
        load_dotenv()
        
        # Check if API key is set
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY is not set in environment variables")
        
        logger.info(f"Initializing NL to SQL converter with model: {model_name}")
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=SQLQueryResult)
    
    def _create_schema_context(self, relevant_schemas: List[Dict[str, Any]]) -> str:
        """
        Create a context string from relevant schema elements.
        
        Args:
            relevant_schemas: List of schema elements
            
        Returns:
            String representation of the schemas
        """
        context = "DATABASE SCHEMA:\n\n"
        
        # Collect unique tables with their content
        tables = {}
        
        for item in relevant_schemas:
            payload = item["payload"]
            payload_type = payload["type"]
            
            # Extract table name consistently
            table_name = payload.get("name", payload.get("table", None))
            
            if not table_name:
                continue
                
            # Skip if we already have this table
            if table_name in tables:
                continue
                
            # For table_with_samples type (our consolidated format)
            if payload_type == "table_with_samples" and "content" in payload:
                tables[table_name] = {"content": payload["content"]}
                
            # For backwards compatibility with older formats
            elif payload_type == "table":
                if "description" in payload and payload["description"]:
                    tables[table_name] = {"description": payload["description"]}
                elif "schema" in payload:
                    tables[table_name] = {"schema": payload["schema"]}
                    
            elif payload_type == "sample_data" and "content" in payload:
                tables[table_name] = {"content": payload["content"]}
        
        # Format schema information
        for table_name, table_info in tables.items():
            if "content" in table_info:
                # Use the pre-formatted content directly
                context += f"{table_info['content']}\n\n"
            elif "description" in table_info:
                # Use pre-formatted description
                context += f"{table_info['description']}\n\n"
            elif "schema" in table_info:
                # Format from schema object
                schema = table_info["schema"]
                context += f"TABLE: {table_name}\n"
                
                # Add columns
                context += "COLUMNS:\n"
                for column in schema.get("columns", []):
                    col_name = column["name"]
                    col_type = column["type"]
                    primary_key = "PRIMARY KEY" if column.get("primary_key", False) else ""
                    nullable = "NULL" if column.get("nullable", True) else "NOT NULL"
                    context += f"  - {col_name}: {col_type} {nullable} {primary_key}\n"
                
                # Add foreign keys if available
                if schema.get("foreign_keys"):
                    context += "FOREIGN KEYS:\n"
                    for fk in schema.get("foreign_keys", []):
                        const_cols = ", ".join(fk.get("constrained_columns", []))
                        ref_table = fk.get("referred_table", "")
                        ref_cols = ", ".join(fk.get("referred_columns", []))
                        context += f"  - ({const_cols}) REFERENCES {ref_table}({ref_cols})\n"
                
                context += "\n"
        
        return context
    
    def nl_to_sql(self, 
                 query: str, 
                 relevant_schemas: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Convert a natural language query to SQL.
        
        Args:
            query: Natural language query
            relevant_schemas: List of relevant schema elements from vector search
            
        Returns:
            Dictionary containing the SQL query and explanation
        """
        try:
            # Create schema context
            schema_context = self._create_schema_context(relevant_schemas)
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_template("""
            You are an expert SQL developer who specializes in converting natural language queries to SQL.
            Your task is to convert the user's question into a valid SQL query based on the provided database schema.

            {schema_context}

            USER QUESTION: {query}

            Generate a SQL query to answer this question. Make sure your query is valid and uses the correct table and column names from the schema.
            If the schema doesn't contain enough information to answer the question directly:
            1. Use exploratory SQL that would help understand the database structure
            2. Explain what additional information might be needed

            {format_instructions}
            """)
            
            # Set format instructions from the parser
            format_instructions = self.parser.get_format_instructions()
            
            # Create chain
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            # Run chain
            result = chain.run(
                schema_context=schema_context,
                query=query,
                format_instructions=format_instructions
            )
            
            # Parse result
            try:
                parsed_result = self.parser.parse(result)
                return {
                    "sql": parsed_result.sql,
                    "explanation": parsed_result.explanation
                }
            except OutputParserException as e:
                logger.error(f"Failed to parse output: {str(e)}")
                
                # Try to extract SQL directly from the response
                if "```sql" in result:
                    sql = result.split("```sql")[1].split("```")[0].strip()
                    return {
                        "sql": sql,
                        "explanation": "Extracted SQL from response."
                    }
                
                return {
                    "sql": "",
                    "explanation": f"Failed to generate SQL: {result}"
                }
                
        except Exception as e:
            logger.error(f"Error in NL to SQL conversion: {str(e)}")
            return {
                "sql": "",
                "explanation": f"Error: {str(e)}"
            } 