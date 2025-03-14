import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy import create_engine, inspect, text, MetaData, Table, Column
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseConnector:
    """
    Manages database connections and schema extraction
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize the database connector.
        
        Args:
            connection_string: Optional SQLAlchemy connection string. If not provided,
                               it will be built from environment variables.
        """
        load_dotenv()
        
        if connection_string:
            self.connection_string = connection_string
        else:
            # Construct from environment variables
            db_host = os.getenv("DB_HOST")
            db_port = os.getenv("DB_PORT")
            db_user = os.getenv("DB_USER")
            db_password = os.getenv("DB_PASSWORD")
            db_name = os.getenv("DB_NAME")
            
            # Default to MySQL but could be parameterized for different DB types
            self.connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        logger.info(f"Connecting to database at {db_host}:{db_port}")
        self.engine = self._create_engine()
        
    def _create_engine(self) -> Engine:
        """Create and return a SQLAlchemy engine"""
        try:
            engine = create_engine(self.connection_string)
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection established successfully")
            return engine
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise
    
    def get_all_table_names(self) -> List[str]:
        """Get all table names from the database"""
        try:
            inspector = inspect(self.engine)
            return inspector.get_table_names()
        except Exception as e:
            logger.error(f"Failed to get table names: {str(e)}")
            return []
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """
        Get detailed schema information for a specific table
        
        Returns:
            Dictionary with table schema information including columns, types, constraints
        """
        try:
            inspector = inspect(self.engine)
            
            # Get column information
            columns = []
            for column in inspector.get_columns(table_name):
                columns.append({
                    "name": column["name"],
                    "type": str(column["type"]),
                    "nullable": column["nullable"],
                    "default": str(column["default"]) if column["default"] else None,
                    "primary_key": column.get("primary_key", False)
                })
            
            # Get foreign key information
            foreign_keys = []
            for fk in inspector.get_foreign_keys(table_name):
                foreign_keys.append({
                    "name": fk.get("name"),
                    "referred_table": fk["referred_table"],
                    "referred_columns": fk["referred_columns"],
                    "constrained_columns": fk["constrained_columns"]
                })
            
            # Get indices
            indices = []
            for index in inspector.get_indexes(table_name):
                indices.append({
                    "name": index["name"],
                    "columns": index["column_names"],
                    "unique": index["unique"]
                })
            
            # Get primary key
            pk = inspector.get_pk_constraint(table_name)
            
            return {
                "name": table_name,
                "columns": columns,
                "primary_key": pk,
                "foreign_keys": foreign_keys,
                "indices": indices
            }
        except Exception as e:
            logger.error(f"Failed to get schema for table {table_name}: {str(e)}")
            return {"name": table_name, "error": str(e)}
    
    def get_full_database_schema(self) -> Dict[str, Dict[str, Any]]:
        """
        Extract the complete database schema
        
        Returns:
            Dictionary mapping table names to their schema information
        """
        tables = self.get_all_table_names()
        schema = {}
        
        for table in tables:
            schema[table] = self.get_table_schema(table)
            
        return schema
    
    def execute_query(self, query: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Execute a SQL query and return the results and column names
        
        Args:
            query: SQL query string to execute
            
        Returns:
            Tuple containing (results as list of dicts, column names list)
        """
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query))
                column_names = result.keys()
                results = [dict(row) for row in result]
                return results, list(column_names)
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise
    
    def get_sample_data(self, table_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get sample data from a table
        
        Args:
            table_name: Name of the table
            limit: Maximum number of rows to return
            
        Returns:
            List of dictionaries containing the sample data
        """
        try:
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            results, _ = self.execute_query(query)
            return results
        except Exception as e:
            logger.error(f"Failed to get sample data for table {table_name}: {str(e)}")
            return [] 