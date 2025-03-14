import os
import re
import gzip
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SQLFileParser:
    """
    Parses a SQL file to extract database schema information and sample data.
    """
    
    def __init__(self, sql_file_path: str):
        """Initialize the SQL file parser."""
        if not os.path.exists(sql_file_path):
            raise FileNotFoundError(f"SQL file not found: {sql_file_path}")
        
        self.sql_file_path = sql_file_path
        self._table_definitions = None
        self._schema_cache = {}
        
        logger.info(f"Initializing SQL file parser for: {sql_file_path}")
    
    def _read_sql_file(self) -> str:
        """Read SQL file content, handling gzipped files."""
        try:
            if self.sql_file_path.endswith('.gz'):
                with gzip.open(self.sql_file_path, 'rt', encoding='utf-8', errors='replace') as f:
                    return f.read()
            else:
                with open(self.sql_file_path, 'r', encoding='utf-8', errors='replace') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Failed to read SQL file: {str(e)}")
            raise
    
    def extract_table_definitions(self) -> Dict[str, str]:
        """Extract CREATE TABLE statements from the SQL file."""
        if self._table_definitions is not None:
            return self._table_definitions
        
        sql_content = self._read_sql_file()
        self._table_definitions = {}
        
        # Extract table definitions with a simpler regex pattern
        pattern = r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:`|\'|")?(\w+)(?:`|\'|")?\s*\(([\s\S]*?)\)([\s\S]*?);'
        matches = re.findall(pattern, sql_content, re.IGNORECASE)
        
        for table_name, columns_part, table_options in matches:
            full_def = f"CREATE TABLE {table_name} ({columns_part}){table_options};"
            self._table_definitions[table_name] = full_def
        
        logger.info(f"Extracted {len(self._table_definitions)} table definitions")
        return self._table_definitions
    
    def get_all_table_names(self) -> List[str]:
        """Get all table names from the SQL file."""
        return list(self.extract_table_definitions().keys())
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Extract detailed schema for a specific table."""
        # Return cached schema if available
        if table_name in self._schema_cache:
            return self._schema_cache[table_name]
            
        table_definitions = self.extract_table_definitions()
        if table_name not in table_definitions:
            logger.error(f"Table {table_name} not found in SQL file")
            return {"name": table_name, "error": "Table not found"}
        
        create_stmt = table_definitions[table_name]
        
        # Extract column definitions section
        col_section_match = re.search(r'CREATE\s+TABLE\s+(?:`|\'|")?(?:\w+)(?:`|\'|")?\s*\(([\s\S]*)\)\s*(?:ENGINE|;)', create_stmt, re.IGNORECASE)
        if not col_section_match:
            return {"name": table_name, "columns": [], "error": "Failed to extract columns"}
        
        column_section = col_section_match.group(1)
        
        # Parse columns and constraints
        columns = []
        constraints = []
        foreign_keys = []
        
        # Split column definitions
        parts = self._split_preserving_parentheses(column_section)
        
        for part in parts:
            part = part.strip()
            # Skip empty parts
            if not part:
                continue
                
            # Check if this is a constraint
            if re.match(r'^(?:PRIMARY|FOREIGN|UNIQUE|KEY|INDEX|CONSTRAINT|CHECK|FULLTEXT)', part.upper()):
                constraints.append(part)
                
                # Extract foreign keys
                fk_match = re.search(r'FOREIGN\s+KEY\s+\(`?([^`)]+)`?\)\s+REFERENCES\s+`?(\w+)`?\s*\(`?([^`)]+)`?\)', part, re.IGNORECASE)
                if fk_match:
                    fk_cols = [c.strip('` \t\n\r') for c in fk_match.group(1).strip().split(',')]
                    ref_table = fk_match.group(2).strip()
                    ref_cols = [c.strip('` \t\n\r') for c in fk_match.group(3).strip().split(',')]
                    
                    foreign_keys.append({
                        "name": f"fk_{table_name}_{ref_table}",
                        "referred_table": ref_table,
                        "referred_columns": ref_cols,
                        "constrained_columns": fk_cols
                    })
            else:
                # Parse column definition
                col_match = re.match(r'`?(\w+)`?\s+(.*)', part)
                if col_match:
                    col_name = col_match.group(1)
                    col_def = col_match.group(2)
                    
                    nullable = "NOT NULL" not in col_def.upper()
                    primary_key = "PRIMARY KEY" in col_def.upper()
                    
                    # Extract default value
                    default_match = re.search(r'DEFAULT\s+([^,\s]+)', col_def, re.IGNORECASE)
                    default = default_match.group(1) if default_match else None
                    
                    # Extract type
                    type_match = re.search(r'(\w+(?:\s*\([^)]*\))?)', col_def)
                    col_type = type_match.group(0) if type_match else "UNKNOWN"
                    
                    columns.append({
                        "name": col_name,
                        "type": col_type,
                        "nullable": nullable,
                        "default": default,
                        "primary_key": primary_key
                    })
        
        # Extract primary key
        pk = {"name": "PRIMARY", "constrained_columns": []}
        for constraint in constraints:
            pk_match = re.search(r'PRIMARY\s+KEY\s+\(`?([^`)]+)`?\)', constraint, re.IGNORECASE)
            if pk_match:
                pk_columns = pk_match.group(1).split(',')
                pk["constrained_columns"] = [col.strip('` \t\n\r') for col in pk_columns]
                break
        
        # If primary key wasn't found in constraints, check column definitions
        if not pk["constrained_columns"]:
            for column in columns:
                if column["primary_key"]:
                    pk["constrained_columns"].append(column["name"])
        
        schema = {
            "name": table_name,
            "columns": columns,
            "primary_key": pk,
            "foreign_keys": foreign_keys
        }
        
        # Cache the schema
        self._schema_cache[table_name] = schema
        return schema
    
    def extract_sample_data(self, table_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Extract sample data from INSERT statements."""
        # Get table schema to know column names
        schema = self.get_table_schema(table_name)
        if not schema or not schema.get("columns"):
            return []
        
        column_names = [col["name"] for col in schema["columns"]]
        
        # Get SQL content
        sql_content = self._read_sql_file()
        
        # Extract INSERT statements
        dump_pattern = rf"-- Dumping data for table `{table_name}`\s*--\s*LOCK TABLES.*?INSERT INTO.*?UNLOCK TABLES"
        dump_match = re.search(dump_pattern, sql_content, re.DOTALL | re.IGNORECASE)
        
        if not dump_match:
            logger.warning(f"No data found for table {table_name}")
            return []
        
        dump_section = dump_match.group(0)
        insert_pattern = rf"INSERT INTO `?{table_name}`?\s+(?:VALUES|VALUE)\s*\(.*?\)(?:\s*,\s*\(.*?\))*\s*;"
        insert_statements = re.findall(insert_pattern, dump_section, re.DOTALL | re.IGNORECASE)
        
        if not insert_statements:
            return []
            
        logger.info(f"Found {len(insert_statements)} INSERT statements for table {table_name}")
        
        # Process INSERT statements to extract rows
        sample_data = []
        rows_extracted = 0
        
        for statement in insert_statements[:limit]:
            # Extract VALUES part
            values_match = re.search(r'VALUES\s*(\(.*\)(?:\s*,\s*\(.*\))*)', statement, re.DOTALL | re.IGNORECASE)
            if not values_match:
                continue
                
            values_str = values_match.group(1)
            
            # Extract rows
            value_sets = re.findall(r'\((.*?)\)', values_str, re.DOTALL)
            
            for value_set in value_sets:
                if rows_extracted >= limit:
                    break
                    
                # Parse values
                values = self._parse_values(value_set)
                
                # Create row dictionary
                row_data = {}
                for i, value in enumerate(values):
                    if i < len(column_names):
                        clean_value = self._clean_value(value)
                        if clean_value is not None:
                            row_data[column_names[i]] = clean_value
                
                if row_data:
                    sample_data.append(row_data)
                    rows_extracted += 1
            
            if rows_extracted >= limit:
                break
        
        return sample_data
    
    def _parse_values(self, values_str: str) -> List[str]:
        """Parse a comma-separated list of SQL values."""
        values = []
        current = ""
        in_quotes = False
        quote_char = None
        escape = False
        paren_level = 0
        
        for char in values_str:
            if escape:
                current += char
                escape = False
            elif char == '\\':
                current += char
                escape = True
            elif char in ["'", '"'] and (not in_quotes or char == quote_char):
                in_quotes = not in_quotes
                quote_char = char if in_quotes else None
                current += char
            elif char == '(' and not in_quotes:
                paren_level += 1
                current += char
            elif char == ')' and not in_quotes:
                paren_level -= 1
                current += char
            elif char == ',' and not in_quotes and paren_level == 0:
                values.append(current.strip())
                current = ""
            else:
                current += char
        
        if current:
            values.append(current.strip())
        
        return values
    
    def _clean_value(self, value: str) -> Any:
        """Clean a SQL value and convert to appropriate type."""
        # Handle NULL or empty values
        if value is None or value.upper() == 'NULL' or value.strip() == '':
            return None
        
        # Handle quoted strings
        if (value.startswith("'") and value.endswith("'")) or \
           (value.startswith('"') and value.endswith('"')):
            # Remove quotes and clean
            value = value[1:-1].replace("\\'", "'").replace('\\"', '"')
            
            # Remove HTML tags
            value = re.sub(r'<[^>]+>', ' ', value)
            
            # Normalize whitespace
            return ' '.join(value.replace('\\r', ' ').replace('\\n', ' ').replace('\\t', ' ').split())
        
        # Handle numbers
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            # If not a number, clean and return
            value = re.sub(r'<[^>]+>', ' ', value)
            return ' '.join(value.replace('\\r', ' ').replace('\\n', ' ').replace('\\t', ' ').split())
    
    def _split_preserving_parentheses(self, text: str) -> List[str]:
        """Split text by commas, preserving nested parentheses."""
        result = []
        level = 0
        current = ""
        
        for char in text:
            if char in '({[':
                level += 1
                current += char
            elif char in ')}]':
                level -= 1
                current += char
            elif char == ',' and level == 0:
                result.append(current.strip())
                current = ""
            else:
                current += char
        
        if current.strip():
            result.append(current.strip())
        
        return result
    
    def get_full_database_schema(self) -> Dict[str, Dict[str, Any]]:
        """Extract the complete database schema."""
        return {table: self.get_table_schema(table) for table in self.get_all_table_names()}
    
    def create_vector_chunks(self, table_name: str, limit: int = 1) -> List[Dict[str, Any]]:
        """Create a single vector chunk for a table with both schema and sample data."""
        # Get table schema
        schema = self.get_table_schema(table_name)
        if not schema:
            return []
            
        # Get sample data (limit to 1 row)
        sample_data = self.extract_sample_data(table_name, limit=1)
        
        # Build a more concise text representation
        content = f"TABLE: {table_name}\n\n"
        
        # Add schema with simplified column representation
        content += "SCHEMA:\n"
        
        # Add primary key information
        pk_columns = schema.get('primary_key', {}).get('constrained_columns', [])
        if pk_columns:
            content += f"PRIMARY KEY: {', '.join(pk_columns)}\n"
        
        # Add foreign keys if available
        if schema.get('foreign_keys'):
            content += "FOREIGN KEYS:\n"
            for fk in schema.get('foreign_keys', []):
                const_cols = ", ".join(fk.get("constrained_columns", []))
                ref_table = fk.get("referred_table", "")
                ref_cols = ", ".join(fk.get("referred_columns", []))
                content += f"  {const_cols} → {ref_table}({ref_cols})\n"
            content += "\n"
        
        # Combine schema and sample data in an integrated format
        content += "COLUMNS WITH SAMPLE DATA:\n"
        
        # First row of sample data if available
        sample_row = sample_data[0] if sample_data else {}
        
        # List all columns with their properties and sample values
        for col in schema.get('columns', []):
            col_name = col['name']
            col_type = col['type']
            
            # Format column attributes
            attrs = []
            if col_name in pk_columns:
                attrs.append("PK")
            if not col.get('nullable', True):
                attrs.append("NOT NULL")
            
            attrs_str = f" ({', '.join(attrs)})" if attrs else ""
            
            # Add sample value if available
            sample_val = ""
            if col_name in sample_row:
                val = sample_row[col_name]
                if isinstance(val, str) and len(val) > 50:
                    val = val[:47] + "..."
                sample_val = f" = {val}"
            
            content += f"  - {col_name}: {col_type}{attrs_str}{sample_val}\n"
        
        # Create chunk
        return [{
            "table_name": table_name,
            "content": content,
            "metadata": {
                "table": table_name,
                "type": "table_with_samples",
                "sample_count": 1 if sample_data else 0
            }
        }] 