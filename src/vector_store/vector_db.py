import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Interface to the Qdrant vector database for storing and querying database schema embeddings.
    """
    
    def __init__(self, 
                 collection_name: str = "db_schema",
                 vector_size: int = 1536):  # default size for OpenAI embeddings
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the collection to store embeddings
            vector_size: Size of embedding vectors (1536 for OpenAI embeddings)
        """
        load_dotenv()
        self.vector_size = vector_size
        self.collection_name = collection_name
        
        # Initialize Qdrant client (local)
        vector_db_path = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
        logger.info(f"Initializing Qdrant client with local path: {vector_db_path}")
        self.client = QdrantClient(path=vector_db_path)
        
        # Track the next available IDs
        self.next_table_id = 1
        self.next_column_id = 10000
        
        # Ensure collection exists
        self._create_collection_if_not_exists()
        
        # Update next IDs based on existing points
        self._update_next_ids()
    
    def _update_next_ids(self) -> None:
        """Update next available IDs based on existing points in the collection"""
        try:
            if self.has_embeddings():
                # Get all points and find the highest IDs
                table_ids = []
                column_ids = []
                
                # Scroll through all points in batches
                offset = None
                limit = 100
                
                while True:
                    result = self.client.scroll(
                        collection_name=self.collection_name,
                        limit=limit,
                        offset=offset
                    )
                    
                    points = result[0]
                    offset = result[1]  # New offset for pagination
                    
                    if not points:
                        break
                    
                    for point in points:
                        if point.payload.get("type") == "table":
                            table_ids.append(point.id)
                        elif point.payload.get("type") == "column":
                            column_ids.append(point.id)
                    
                    if offset is None:
                        break
                
                if table_ids:
                    self.next_table_id = max(table_ids) + 1
                    logger.info(f"Next table ID set to {self.next_table_id}")
                
                if column_ids:
                    self.next_column_id = max(column_ids) + 1
                    logger.info(f"Next column ID set to {self.next_column_id}")
        except Exception as e:
            logger.error(f"Failed to update next IDs: {str(e)}")
    
    def _create_collection_if_not_exists(self) -> bool:
        """
        Create the vector collection if it doesn't exist already
        
        Returns:
            bool: True if collection already existed, False if it was created
        """
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
                )
                return False
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                return True
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            raise
    
    def has_embeddings(self) -> bool:
        """
        Check if the collection already has embeddings stored
        
        Returns:
            bool: True if collection has data, False otherwise
        """
        try:
            # Try to get collection info
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            
            # Check if collection has points
            if collection_info.points_count > 0:
                logger.info(f"Collection {self.collection_name} already has {collection_info.points_count} embeddings")
                return True
            else:
                logger.info(f"Collection {self.collection_name} exists but is empty")
                return False
        except Exception as e:
            logger.error(f"Failed to check if collection has embeddings: {str(e)}")
            return False
    
    def store_table_embeddings(self, embeddings_data: Dict[str, Dict[str, Any]]) -> None:
        """
        Store table embeddings in the vector database.
        
        Args:
            embeddings_data: Dictionary containing embeddings for tables
        """
        try:
            # Build points for each table
            points = []
            
            for table_name, table_data in embeddings_data.items():
                # Store table embedding
                table_point = models.PointStruct(
                    id=self.next_table_id,
                    vector=table_data["table_embedding"],
                    payload={
                        "type": "table",
                        "name": table_name,
                        "schema": table_data["schema"],
                        "description": table_data.get("description", "")  # Store description for reference
                    }
                )
                points.append(table_point)
                self.next_table_id += 1
            
            # Upsert points in batches (100 at a time)
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                logger.info(f"Stored batch of {len(batch)} embeddings")
            
            logger.info(f"Successfully stored {len(points)} table embeddings")
        except Exception as e:
            logger.error(f"Failed to store embeddings: {str(e)}")
            raise
    
    def search_schema(self, 
                      query_embedding: List[float], 
                      limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for schema elements matching a query embedding.
        
        Args:
            query_embedding: Embedding vector of the query
            limit: Maximum number of results to return
            
        Returns:
            List of matching schema elements with scores
        """
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )
            
            results = []
            for scored_point in search_result:
                results.append({
                    "id": scored_point.id,
                    "score": scored_point.score,
                    "payload": scored_point.payload
                })
            
            return results
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
    
    def get_all_tables(self) -> List[Dict[str, Any]]:
        """
        Retrieve all tables from the vector store.
        
        Returns:
            List of table schemas
        """
        try:
            # Filter to get only table entries
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="type",
                        match=models.MatchValue(value="table")
                    )
                ]
            )
            
            result = self.client.scroll(
                collection_name=self.collection_name,
                filter=filter_condition,
                limit=1000  # Assuming won't have more than 1000 tables
            )
            
            tables = []
            for point in result[0]:
                tables.append(point.payload)
            
            return tables
        except Exception as e:
            logger.error(f"Failed to get all tables: {str(e)}")
            return []
    
    def clear_collection(self) -> None:
        """Clear all data from the collection"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self._create_collection_if_not_exists()
            
            # Reset IDs
            self.next_table_id = 1
            self.next_column_id = 10000
            
            logger.info(f"Successfully cleared collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
            raise
            
    def store_embeddings(self, embeddings_list: List[Dict[str, Any]]) -> None:
        """
        Store generic embeddings in the vector database.
        Can be used for sample data or other types of embeddings.
        
        Args:
            embeddings_list: List of dictionaries containing vector and payload
                            Each dict should have "vector" and "payload" keys
        """
        try:
            if not embeddings_list:
                logger.warning("No embeddings provided to store")
                return
                
            # Determine the starting ID based on payload type
            next_id = self.next_table_id + 1000  # Use a different range for generic embeddings
            
            # Build points
            points = []
            for embedding_data in embeddings_list:
                vector = embedding_data.get("vector")
                payload = embedding_data.get("payload", {})
                
                if not vector:
                    logger.warning(f"Skipping embedding with no vector: {payload}")
                    continue
                    
                # Create point
                point = models.PointStruct(
                    id=next_id,
                    vector=vector,
                    payload=payload
                )
                points.append(point)
                next_id += 1
            
            # Upsert points in batches (100 at a time)
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                logger.info(f"Stored batch of {len(batch)} embeddings")
            
            logger.info(f"Successfully stored {len(points)} embeddings")
        except Exception as e:
            logger.error(f"Failed to store generic embeddings: {str(e)}")
            raise 