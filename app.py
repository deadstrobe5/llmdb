#!/usr/bin/env python3
import streamlit as st
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Suppress excessive logging
logging.basicConfig(level=logging.ERROR)
for logger_name in ['httpx', 'src.db.sql_file_parser', 'src.vector_store.embeddings']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Add the project directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.vector_store.embeddings import SchemaEmbedder
from src.vector_store.vector_db import VectorStore
from src.llm.nl_to_sql import NLToSQL

# Initialize components (do this only once)
@st.cache_resource
def load_components():
    load_dotenv()
    embedder = SchemaEmbedder()
    vector_store = VectorStore()
    nl_to_sql = NLToSQL()
    
    if not vector_store.has_embeddings():
        st.error("Vector database has no embeddings! Run init_vector_db.py first.")
        st.stop()
        
    return embedder, vector_store, nl_to_sql

# Page setup
st.set_page_config(
    page_title="NL to SQL Converter",
    page_icon="🔍",
    layout="wide"
)

st.title("💬 Natural Language to SQL Converter")
st.markdown("Ask questions about your database in plain English and get SQL queries.")

# Load components
embedder, vector_store, nl_to_sql = load_components()

# Input section
query = st.text_area("Enter your natural language query:", height=100, 
                    placeholder="Example: Find all users with admin role")

# Submit button
if st.button("Generate SQL", type="primary"):
    if not query:
        st.warning("Please enter a query")
    else:
        with st.spinner("Generating SQL query..."):
            # Get query embedding
            query_embedding = embedder.embed_query(query)
            
            # Search for relevant schemas
            relevant_schemas = vector_store.search_schema(query_embedding, limit=5)
            
            if not relevant_schemas:
                st.error("No relevant schema information found")
                st.stop()
            
            # Convert to SQL
            result = nl_to_sql.nl_to_sql(query, relevant_schemas)
            
            if result and result["sql"]:
                # Display SQL with syntax highlighting
                st.subheader("Generated SQL")
                st.code(result["sql"], language="sql")
                
                # Display explanation
                st.subheader("Explanation")
                st.write(result["explanation"])
                
                # Show relevant schema info in an expandable section
                with st.expander("View schema information used"):
                    for i, schema in enumerate(relevant_schemas[:3]):
                        payload = schema['payload']
                        st.markdown(f"**Table: {payload.get('name', payload.get('table', 'Unknown'))}** (score: {schema['score']:.4f})")
                        
                        # Only show the first part of the content to keep it clean
                        if 'content' in payload:
                            content_preview = payload['content'].split("\n\n")[0]
                            st.text(content_preview)
                        st.divider()
            else:
                st.error("Failed to generate SQL query")

# Add some helpful examples
with st.sidebar:
    st.header("Example Queries")
    examples = [
        "Show me all users",
        "Find content created by user 'plus'",
        "List users with names containing the letter 'd'",
        "Show all categories with their parent categories",
        "Find the most recent content items",
        "List all tags used more than 5 times"
    ]
    
    for example in examples:
        if st.button(example):
            st.session_state.query = example
            st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Natural Language to SQL with vector embeddings") 